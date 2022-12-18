import torch.multiprocessing

from util.distil_utils import calc_direction_split, compute_offsets

torch.multiprocessing.set_sharing_strategy('file_system')    # a fix for the "OSError: too many files" exception

import math
import shutil
from collections import OrderedDict
from pathlib import Path

import hydra
import lpips
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
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
from model.generator import Generator, SynthesisNetwork
from model.loss import PathLengthPenalty, compute_gradient_penalty
from trainer import create_trainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


def process_generator_ckpt(ckpt) -> OrderedDict:
    processed_state_dict = OrderedDict()

    for w_name, w in ckpt['state_dict'].items():
        if w_name[0] == 'G':
            processed_state_dict[w_name[2:]] = w

    return processed_state_dict


class StyleGAN2Module(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # wget https://www.dropbox.com/s/2ovhzt1rzhazfyq/stylegan2_ada_lightning_epoch_110.ckpt
        ckpt = torch.load('stylegan2_ada_lightning_epoch_110.ckpt')
        generator_state_dict = process_generator_ckpt(ckpt)

        self.teacher_generator = Generator(
            512, 512,
            w_num_layers=2,
            img_resolution=32,
            img_channels=3,
            synthesis_layer='stylegan2',
        )
        self.teacher_generator.load_state_dict(generator_state_dict)

        if config.use_simi_loss:
            if config.mimin_layers is None or len(config.mimin_layers) < 1:
                raise ValueError

            self.offset_vectors = calc_direction_split(self.teacher_generator, config.mimin_layers)

        self.recover_net0 = \
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ).to(self.device)
        self.recover_net1 = \
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ).to(self.device)
        self.recover_net2 = \
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ).to(self.device)

        self.G = Generator(
            config.latent_dim,
            config.latent_dim,
            config.num_mapping_layers,
            config.image_size,
            img_channels=3,
            synthesis_layer=config.generator,
        )
        self.G.load_state_dict(generator_state_dict)
        for param in self.G.parameters():
            param.requires_grad = False
        self.G.synthesis = SynthesisNetwork(
            w_dim=config.latent_dim,
            img_resolution=32,
            img_channels=3,
            channel_max=256, #twice less than in teacher's network1
            synthesis_layer='stylegan2',
        )

        self.D = Discriminator(config.image_size, 3)
        self.augment_pipe = AugmentPipe(
            config.ada_start_p,
            config.ada_target,
            config.ada_interval,
            config.ada_fixed,
            config.batch_size,
        )
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

        self.rgb_criterion = nn.L1Loss()
        self.lpips_criterion = lpips.LPIPS(net='vgg')

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.G.parameters()), lr=self.config.lr_g, betas=(0.0, 0.99), eps=1e-8)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8)
        return g_opt, d_opt

    def forward(self, limit_batch_size=False, return_features=False):
        z = self.latent(limit_batch_size)
        w = self.get_mapped_latent(z, 0.9)
        if return_features:
            fake, features = self.G.synthesis(w, return_features=True)
            return fake, w, features
        else:
            fake = self.G.synthesis(w, return_features=False)
            return fake, w

    def corrupt(self, feature):
        bs, _, h, w = feature.shape
        mask = torch.rand((bs, 1, h, w), device=self.device)
        mask = torch.where(mask > 0.8, 0, 1)
        corrupted_feature = mask * feature
        return corrupted_feature

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

        g_opt.zero_grad(set_to_none=True)
        log_rgb_loss = \
            torch.tensor(0, dtype=torch.float32, device=self.device)
        for acc_step in range(total_acc_steps):
            fake, w = self.forward()
            teacher_fake = self.teacher_generator.synthesis(w)
            loss_rgb0 = self.rgb_criterion(fake, teacher_fake)
            loss_rgb = self.config.rgb_coef * loss_rgb0
            self.manual_backward(loss_rgb)
            log_rgb_loss += loss_rgb.detach()
        g_opt.step()
        log_rgb_loss /= total_acc_steps
        self.log("rgb_loss", log_rgb_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        g_opt.zero_grad(set_to_none=True)
        log_lpips_loss = \
            torch.tensor(0, dtype=torch.float32, device=self.device)
        for acc_step in range(total_acc_steps):
            fake, w = self.forward()
            teacher_fake = self.teacher_generator.synthesis(w)

            loss_lpips0 = self.lpips_criterion(fake, teacher_fake).mean()
            loss_lpips = self.config.lpips_coef * loss_lpips0
            self.manual_backward(loss_lpips)
            log_lpips_loss += loss_lpips.detach()
        g_opt.step()
        log_lpips_loss /= total_acc_steps
        self.log("lpips_loss", log_lpips_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        if self.config.use_mgd_loss:
            g_opt.zero_grad(set_to_none=True)
            _, w, student_features = self.forward(return_features=True)
            _, teacher_features = \
                self.teacher_generator.synthesis(w, return_features=True)

            mgd_loss = 0
            for index in self.config.mgd_layers:
                t1 = teacher_features[index]
                s1 = student_features[index]
                corrupted_s1 = self.corrupt(s1)
                if index == 0:
                    recover_net = self.recover_net0
                elif index == 1:
                    recover_net = self.recover_net1
                elif index == 2:
                    recover_net = self.recover_net2
                rec_s1 = recover_net(corrupted_s1)
                mgd_loss += F.mse_loss(rec_s1, t1)

            mgd_loss *= 0.05
            self.manual_backward(mgd_loss)
            g_opt.step()
            self.log("mgd_loss", mgd_loss.detach(), on_step=True, on_epoch=False,
                     prog_bar=False, logger=True, sync_dist=True)
            # torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        if self.config.use_simi_loss:
            g_opt.zero_grad(set_to_none=True)
            _, w, student_features = self.forward(return_features=True)

            offsets = compute_offsets(self.offset_vectors, self.config.offset_weight, w.shape[0], self.device)
            w_offset = w + offsets

            _, student_features_offset = self.G.synthesis(w_offset, return_features=True)

            _, teacher_features = self.teacher_generator.synthesis(w, return_features=True)
            _, teacher_features_offset = self.teacher_generator.synthesis(w_offset, return_features=True)

            simi_loss = 0
            for index in self.config.mimin_layers:
                f1 = student_features[index]
                f2 = student_features_offset[index]
                s_simi = F.cosine_similarity(f1[:, None, :], f2[None, :, :], dim=2)

                f1 = teacher_features[index]
                f2 = teacher_features_offset[index]
                t_simi = F.cosine_similarity(f1[:, None, :], f2[None, :, :], dim=2)

                simi_loss += F.mse_loss(s_simi, t_simi)

            simi_loss *= self.config.simi_lambda

            self.manual_backward(simi_loss)
            g_opt.step()

            self.log("simi_loss", simi_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, logger=True,
                     sync_dist=True)

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
    model = StyleGAN2Module(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()

# LATENT:
# noise - [(b, 512), (b, 512)] # https://github.com/xuguodong03/StyleKD/blob/6ae4d1724d1b4a2df4e987c9aec0f51c27c1e8d2/distributed_train.py#L210
# styles = [style(s) for s in noise]
# latent = b, 1, 512 -> b, 18, 512
# latent = torch.cat([latent, latent + offset], dim=0) 2 * b, 18, 512 https://github.com/xuguodong03/StyleKD/blob/master/model.py#L618

# LOSS
# l1_loss = torch.mean(torch.abs(fake_teacher - fake_student))
# style_loss = l1(student_w, teacher_w)
# perceptual_loss/lpips - vgg - torch.mean(percept_loss(fake, real))
# mimic_layer = [2, 3, 4, 5], feature - [(B, -1), (), ...]
# cos_sim(feat_student[b, 1, h]{latent}, feat_student[1, b, h]{latent+offset}, dim=2)
# mse(cos_sim(student, student), cos_sim(teacher, teacher))


# ??? parsing net https://github.com/xuguodong03/StyleKD/blob/6ae4d1724d1b4a2df4e987c9aec0f51c27c1e8d2/distributed_train.py#L546
# single_view тут по def_arg false https://github.com/xuguodong03/StyleKD/blob/6ae4d1724d1b4a2df4e987c9aec0f51c27c1e8d2/distributed_train.py#L134
# simi_loss по def_arg mse