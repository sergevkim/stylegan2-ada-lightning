from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size, size=None):
        super().__init__()
        self.paths = [p for p in Path(path).iterdir() if p.name.endswith('.png') or p.name.endswith('.jpg')]
        self.size = min(size if size is not None else len(self.paths), len(self.paths))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return {
            'image': self.transform(img) * 2 - 1
        }


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_cifar, # CIFAR10 or subset of it
    ) -> None:
        super().__init__()
        self.raw_cifar = raw_cifar

    def __len__(self) -> int:
        return len(self.raw_cifar)

    def __getitem__(self, idx):
        image, label = self.raw_cifar[idx]
        # be careful! image is not a numpy array

        return {
            'image': image,
            'label': label,
        }