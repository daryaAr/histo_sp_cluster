import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import timm
from pathlib import Path
from torchvision import transforms
from histo_MIL.config import load_yaml_config
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from histo_MIL.get_embeddings.get_model_inference import get_model


def get_encoder(cfg, checkpoint_path, model):
    device = cfg.training.device
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    encoder = model.encoder_q
    encoder.fc = nn.Identity()
    encoder.eval()
    return encoder

def get_embeddings(cfg, json_path, model, checkpoint_path, data_type="train", batch_size=512, num_workers=32):
    device = torch.device(cfg.training.device)

    # ----- Load JSON -----
    with open(json_path, "r") as f:
        data = json.load(f)

    tile_paths, tile_info = [], []
    for entry in data:
        wsi_id = entry["wsi_id"]
        label = entry["label"]
        for tile_path in entry["tiles_files"]:
            tile_paths.append(tile_path)
            tile_info.append((wsi_id, label, tile_path))

    # ----- Dataset -----
    class TileDataset(Dataset):
        def __init__(self, tile_paths):
            self.tile_paths = tile_paths
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517)
                )
            ])

        def __len__(self):
            return len(self.tile_paths)

        def __getitem__(self, idx):
            img_path = self.tile_paths[idx]
            image = Image.open(img_path).convert("RGB")
            return self.transform(image), img_path

    dataset = TileDataset(tile_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ----- Load model -----
    
    encoder = get_encoder(cfg, checkpoint_path, model)
    # ----- Extract embeddings -----
    all_embeddings, all_paths = [], []
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc=f"Extracting embeddings from {data_type}"):
            imgs = imgs.to(device)
            feats = encoder(imgs)
            all_embeddings.append(feats.cpu())
            all_paths.extend(paths)

    result = {
        "embeddings": torch.cat(all_embeddings),
        "paths": all_paths,
        "tile_info": tile_info
    }

    # ----- Save -----
    save_dir = Path(cfg.cptac.embeddings_result) / "mocov2"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{data_type}.pt"
    torch.save(result, save_path)
    print(f"✅ Saved embeddings of {data_type} to {save_path}")

    return result

if __name__ == "__main__":
    cfg = load_yaml_config()
    checkpoint_path = Path(cfg.paths.checkpoint_dir)/"no_cluster_bs256_lr0.003_step50_warmup100_epochs100_clusters50_moco_v2"/ "best_model.pth"
    model = get_model(cfg, model_type="moco_v2", inference_only=False)
    train_json = Path(cfg.cptac.json_output) / "updated_train.json"
    get_embeddings(
        cfg=cfg,
        json_path=train_json,
        model=model,
        checkpoint_path=checkpoint_path,
        data_type="train"
    )
    test_json = Path(cfg.cptac.json_output) / "updated_test.json"
    get_embeddings(
        cfg=cfg,
        json_path=test_json,
        model=model,
        checkpoint_path=checkpoint_path,
        data_type="test"
    ) 