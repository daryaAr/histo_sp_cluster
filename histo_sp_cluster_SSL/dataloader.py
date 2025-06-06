from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import random
import json
import pyspng
import torch


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

"""
def get_moco_v2_augmentations():
    
    Returns the MoCo v2 augmentation pipeline.
    
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(sigma=[0.1, 2.0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
"""

def get_histo_moco_augmentations():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
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

class SuperpixelMoCoDatasetNeighborCluster(Dataset):
    """
    Dataset for MoCo training using superpixels, returns images and their cluster info.
    Each item returns:
        (anchor_1, anchor_2, neighbor, anchor_clusters, neighbor_clusters)
    """

    def __init__(self, cfg, transform=None):
        with open(cfg.paths.bioptimus_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def _load_image(self, path):
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        return Image.fromarray(img).convert("RGB")

    def __getitem__(self, idx):
        group = self.superpixel_list[idx]
        tile_entries = group["tiles"]

        anchor = random.choice(tile_entries)
        neighbor = random.choice(tile_entries)

        anchor_img = self._load_image(anchor["tile_path"])
        neighbor_img = self._load_image(neighbor["tile_path"])

        if self.transform:
            img_q = self.transform(anchor_img)
            img_k1 = self.transform(anchor_img)
            img_k2 = self.transform(neighbor_img)

        anchor_clusters = (
            anchor["primary_cluster"],
            anchor["second_cluster"],
            anchor["third_cluster"]
        )
        neighbor_clusters = (
            neighbor["primary_cluster"],
            neighbor["second_cluster"],
            neighbor["third_cluster"]
        )

        return img_q, img_k1, img_k2, torch.tensor(anchor_clusters, dtype=torch.long), torch.tensor(neighbor_clusters, dtype=torch.long)        


def get_dataloader(cfg, use_neighbors=False):
    """
    Returns a PyTorch DataLoader for MoCo training.
    """
    #dataset_cls = SuperpixelMoCoDatasetNeighbor if use_neighbors else SuperpixelMoCoDataset
    if use_neighbors:
        if cfg.cluster.cluster_type == "cluster_bioptimus":
            dataset_cls = SuperpixelMoCoDatasetNeighborCluster
        else:
            dataset_cls = SuperpixelMoCoDatasetNeighbor
    else:
        dataset_cls = SuperpixelMoCoDataset




    transform = get_histo_moco_augmentations()
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