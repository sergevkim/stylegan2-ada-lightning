import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')    # a fix for the "OSError: too many files" exception

import math
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from cleanfid import fid
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from dataset.image import CIFAR10Dataset
from model.augment import AugmentPipe
from model.discriminator import Discriminator
from model.generator import Generator
from model.loss import PathLengthPenalty, compute_gradient_penalty
from trainer import create_trainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class StyleGAN2Trainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.G = Generator(
            config.latent_dim,
            config.latent_dim,
            config.num_mapping_layers,
            config.image_size,
            img_channels=3,
            synthesis_layer=config.generator,
        )
        self.D = Discriminator(config.image_size, 3)
        self.augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size)
        self.grid_z = torch.randn(config.num_eval_images, self.config.latent_dim)
        raw_train_cifar = CIFAR10(
            config.dataset_path,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        raw_val_cifar = CIFAR10(
            config.dataset_path,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        self.train_set = CIFAR10Dataset(raw_train_cifar)
        self.val_set = CIFAR10Dataset(raw_val_cifar)
        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.ema = None

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.G.parameters()), lr=self.config.lr_g, betas=(0.0, 0.99), eps=1e-8)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8)
        return g_opt, d_opt

    def forward(self, limit_batch_size=False):
        z = self.latent(limit_batch_size)
        w = self.get_mapped_latent(z, 0.9)
        fake = self.G.synthesis(w)
        return fake, w

    def training_step(self, batch, batch_idx):
        total_acc_steps = self.config.batch_size // self.config.batch_gpu
        g_opt, d_opt = self.optimizers()

        # optimize generator
        g_opt.zero_grad(set_to_none=True)
        log_gen_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for acc_step in range(total_acc_steps):
            fake, w = self.forward()
            p_fake = self.D(self.augment_pipe(fake))
            gen_loss = torch.nn.functional.softplus(-p_fake).mean()
            self.manual_backward(gen_loss)
            log_gen_loss += gen_loss.detach()
        g_opt.step()
        log_gen_loss /= total_acc_steps
        self.log("G", log_gen_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        if self.global_step > self.config.lazy_path_penalty_after and (self.global_step + 1) % self.config.lazy_path_penalty_interval == 0:
            g_opt.zero_grad(set_to_none=True)
            log_plp_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            for acc_step in range(total_acc_steps):
                fake, w = self.forward()
                plp = self.path_length_penalty(fake, w)
                if not torch.isnan(plp):
                    plp_loss = self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
                    self.manual_backward(plp_loss)
                    log_plp_loss += plp.detach()
            g_opt.step()
            log_plp_loss /= total_acc_steps
            self.log("rPLP", log_plp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        # optimize discriminator
        d_opt.zero_grad(set_to_none=True)
        log_real_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_fake_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        batch_image = batch["image"].split(self.config.batch_gpu)
        for acc_step in range(total_acc_steps):
            fake, _ = self.forward()
            p_fake = self.D(self.augment_pipe(fake.detach()))
            fake_loss = torch.nn.functional.softplus(p_fake).mean()
            self.manual_backward(fake_loss)
            log_fake_loss += fake_loss.detach()

            p_real = self.D(self.augment_pipe(batch_image[acc_step]))
            self.augment_pipe.accumulate_real_sign(p_real.sign().detach())
            real_loss = torch.nn.functional.softplus(-p_real).mean()
            self.manual_backward(real_loss)
            log_real_loss += real_loss.detach()

        d_opt.step()
        log_real_loss /= total_acc_steps
        log_fake_loss /= total_acc_steps
        self.log("D_real", log_real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("D_fake", log_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        disc_loss = log_real_loss + log_fake_loss
        self.log("D", disc_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        if (self.global_step + 1) % self.config.lazy_gradient_penalty_interval == 0:
            d_opt.zero_grad(set_to_none=True)
            log_gp_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            batch_image = batch["image"].split(self.config.batch_gpu)
            for acc_step in range(total_acc_steps):
                batch_image[acc_step].requires_grad_(True)
                p_real = self.D(self.augment_pipe(batch_image[acc_step], disable_grid_sampling=True))
                gp = compute_gradient_penalty(batch_image[acc_step], p_real)
                gp_loss = self.config.lambda_gp * gp * self.config.lazy_gradient_penalty_interval
                self.manual_backward(gp_loss)
                log_gp_loss += gp.detach()
            d_opt.step()
            log_gp_loss /= total_acc_steps
            self.log("rGP", log_gp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        self.execute_ada_heuristics()
        self.ema.update(self.G.parameters())

    def execute_ada_heuristics(self):
        if (self.global_step + 1) % self.config.ada_interval == 0:
            self.augment_pipe.heuristic_update()
        self.log("aug_p", self.augment_pipe.p.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pass

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        odir_real, odir_fake, odir_samples = self.create_directories()
        self.export_images("", odir_samples, None)
        self.ema.store(self.G.parameters())
        self.ema.copy_to(self.G.parameters())
        self.export_images("ema_", odir_samples, odir_fake)
        self.ema.restore(self.G.parameters())
        for iter_idx, batch in enumerate(self.val_dataloader()):
            for batch_idx in range(batch['image'].shape[0]):
                save_image(batch['image'][batch_idx], odir_real / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
        fid_score = \
            fid.compute_fid(str(odir_real), str(odir_fake), device=self.device)
        kid_score = \
            fid.compute_kid(str(odir_real), str(odir_fake), device=self.device)
        self.log(f"fid", fid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        self.log(f"kid", kid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        print(f'FID: {fid_score:.3f} , KID: {kid_score:.3f}')
        shutil.rmtree(odir_real.parent)

    def get_mapped_latent(self, z, style_mixing_prob):
        if torch.rand(()).item() < style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.G.mapping.num_ws)
            w1 = self.G.mapping(z[0])[:, :cross_over_point, :]
            w2 = self.G.mapping(z[1], skip_w_avg_update=True)[:, cross_over_point:, :]
            return torch.cat((w1, w2), dim=1)
        else:
            w = self.G.mapping(z[0])
            return w

    def latent(self, limit_batch_size=False):
        batch_size = self.config.batch_gpu if not limit_batch_size else self.config.batch_gpu // self.path_length_penalty.pl_batch_shrink
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.config.batch_gpu,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
        )

    def export_images(self, prefix, output_dir_vis, output_dir_fid):
        vis_generated_images = []
        for iter_idx, latent in enumerate(self.grid_z.split(self.config.batch_gpu)):
            latent = latent.to(self.device)
            fake = self.G(latent, noise_mode='const').cpu()
            if output_dir_fid is not None:
                for batch_idx in range(fake.shape[0]):
                    save_image(fake[batch_idx], output_dir_fid / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
            if iter_idx < self.config.num_vis_images // self.config.batch_gpu:
                vis_generated_images.append(fake)
        torch.cuda.empty_cache()
        vis_generated_images = torch.cat(vis_generated_images, dim=0)

        # make grid
        grid = torchvision.utils.make_grid(
            vis_generated_images,
            nrow=int(math.sqrt(vis_generated_images.shape[0])),
            value_range=(-1, 1),
            normalize=True,
        )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        grid_image = Image.fromarray(ndarr)
        self.logger.experiment.log({'fake_images': [wandb.Image(grid_image)]})

        save_image(
            vis_generated_images,
            output_dir_vis / f"{prefix}{self.global_step:06d}.png",
            nrow=int(math.sqrt(vis_generated_images.shape[0])),
            value_range=(-1, 1),
            normalize=True,
        )

    def create_directories(self):
        output_dir_fid_real = Path(f'runs/{self.config.experiment}/fid/real')
        output_dir_fid_fake = Path(f'runs/{self.config.experiment}/fid/fake')
        output_dir_fid_samples = Path(f'runs/{self.config.experiment}/images/')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_fid_samples]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_fid_samples

    def on_train_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)

    def on_validation_start(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)


@hydra.main(config_path='./config', config_name='stylegan2')
def main(config):
    trainer = create_trainer("StyleGAN2", config)
    model = StyleGAN2Trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
