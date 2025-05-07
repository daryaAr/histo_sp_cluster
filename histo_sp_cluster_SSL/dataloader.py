from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import random
import json
import pyspng


class TwoCropsTransform:
    """
    Applies two independently augmented views of the same image.
    Used for contrastive learning (e.g., MoCo).
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class GaussianBlur:
    """
    Applies Gaussian blur with a random sigma.
    Used in SimCLR and MoCo v2.
    """
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def get_moco_v2_augmentations():
    """
    Returns the MoCo v2 augmentation pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(sigma=[0.1, 2.0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class SuperpixelMoCoDataset(Dataset):
    """
    Dataset for self-supervised MoCo training using superpixel groupings.
    Each item returns two augmentations of a randomly sampled patch from the group.
    """
    def __init__(self, cfg, transform=None):
        with open(cfg.paths.json_path, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def _load_image(self, path):
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        return Image.fromarray(img).convert("RGB")

    def __getitem__(self, idx):
        tile_paths = self.superpixel_list[idx]["tile_paths"]
        tile_path = random.choice(tile_paths)
        img = self._load_image(tile_path)

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return img1, img2


class SuperpixelMoCoDatasetNeighbor(Dataset):
    """
    Dataset for neighborhood-aware MoCo training.
    Returns (anchor_1, anchor_2, neighbor) for each patch.
    """
    def __init__(self, cfg, transform=None):
        with open(cfg.paths.json_path, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def _load_image(self, path):
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        return Image.fromarray(img).convert("RGB")

    def __getitem__(self, idx):
        tile_paths = self.superpixel_list[idx]["tile_paths"]

        anchor_path = random.choice(tile_paths)
        neighbor_path = random.choice(tile_paths)

        anchor_img = self._load_image(anchor_path)
        neighbor_img = self._load_image(neighbor_path)

        if self.transform:
            anchor_1 = self.transform(anchor_img)
            anchor_2 = self.transform(anchor_img)
            neighbor = self.transform(neighbor_img)

        return anchor_1, anchor_2, neighbor


def get_dataloader(cfg, use_neighbors=False):
    """
    Returns a PyTorch DataLoader for MoCo training.
    """
    dataset_cls = SuperpixelMoCoDatasetNeighbor if use_neighbors else SuperpixelMoCoDataset
    transform = get_moco_v2_augmentations()
    dataset = dataset_cls(cfg, transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor,
    )

    return train_loader