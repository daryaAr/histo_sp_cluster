import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import timm
from pathlib import Path
from torchvision import models
from torchvision import transforms
from histo_MIL.config import load_yaml_config
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_resnet_embeddings(cfg, json_path, batch_size=64, num_workers=4, data_type="train"):
    """
    Extracts ResNet50 (ImageNet pretrained) embeddings for tiles listed in a JSON file.

    Args:
        json_path (str or Path): Path to the JSON file.
        save_path (str or Path): If provided, saves the output as a .pt file.
        batch_size (int): Batch size for dataloader.
        num_workers (int): Number of workers for dataloader.
        device (str or torch.device): Device to run the model on.

    Returns:
        dict: {
            "embeddings": Tensor of shape (N, 2048),
            "paths": List of tile paths,
            "tile_info": List of (wsi_id, label, tile_path)
        }
    """
    device = cfg.training.device
     

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    tile_paths, tile_info = [], []
    for entry in data:
        wsi_id = entry["wsi_id"]
        label = entry["label"]
        for tile_path in entry["tiles_files"]:
            tile_paths.append(tile_path)
            tile_info.append((wsi_id, label, tile_path))

    # Dataset
    class TileDataset(Dataset):
        def __init__(self, tile_paths, transform=None):
            self.tile_paths = tile_paths
            self.transform = transform

        def __len__(self):
            return len(self.tile_paths)

        def __getitem__(self, idx):
            img_path = self.tile_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    dataset = TileDataset(tile_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load pretrained ResNet50
    model = getattr(models, "resnet50")(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    # Extract embeddings
    all_embeddings, all_paths = [], []
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc=f"Extracting embeddings from {data_type}"):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_embeddings.append(feats.cpu())
            all_paths.extend(paths)

    result = {
        "embeddings": torch.cat(all_embeddings),
        "paths": all_paths,
        "tile_info": tile_info
    }

    os.makedirs(cfg.cptac.embeddings_result, exist_ok=True)
    torch.save(result, os.path.join(cfg.cptac.embeddings_result, "resnet", f"{data_type}_resnet_embeddings.pt"))
    print(f"✅ Saved embeddings of {data_type} to {cfg.cptac.resnet_embeddings}")

    return result


if __name__ == "__main__":
    cfg = load_yaml_config()
    train_json = Path(cfg.cptac.json_output) / "train.json"
    get_resnet_embeddings(cfg, train_json, batch_size=64, num_workers=4, data_type="train")
    test_json = Path(cfg.cptac.json_output) / "test.json"
    get_resnet_embeddings(cfg, test_json, batch_size=64, num_workers=4, data_type="test")    