import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
import timm
from pathlib import Path
from torchvision import transforms
from histo_MIL.config import load_yaml_config

def load_image_tensor(path, transform, device):
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def extract_bioptimus_embeddings(cfg, json_path, data= "train"):
    login(token=os.getenv("BIOPTIMUS_TOKEN"))

    # Load model
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )
    model.to(cfg.training.device)
    model.eval()

    # Normalization transform (no augmentations)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        )
    ])

    # Load tile paths from JSON
    with open(json_path, "r") as f:
        metadata = json.load(f)

    tile_paths = []
    for entry in metadata:
        tile_paths.extend(entry["tiles_files"]) 

    all_embeddings = []
    model_dtype = torch.float16  # for speed

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=model_dtype):
        for tile_path in tqdm(tile_paths, desc="Extracting Bioptimus Embeddings"):
            try:
                img_tensor = load_image_tensor(tile_path, transform, cfg.training.device)
                features = model(img_tensor)
                all_embeddings.append(features.cpu())
            except Exception as e:
                print(f"Failed to process {tile_path}: {e}")

    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    # Save
    os.makedirs(cfg.cptac.bioptimus_embeddings, exist_ok=True)
    torch.save(embeddings_tensor, os.path.join(cfg.cptac.bioptimus_embeddings, f"{data}_bioptimus_embeddings.pt"))
    torch.save(tile_paths, os.path.join(cfg.cptac.bioptimus_embeddings, f"{data}_tile_paths.pt"))
    print(f"Saved embeddings and tile paths of {data} to {cfg.cptac.bioptimus_embeddings}")

if __name__ == "__main__":
    cfg = load_yaml_config()
    train_json = Path(cfg.cptac.json_output) / "train.json"
    #extract_bioptimus_embeddings(cfg, train_json, data = "train")
    test_json = Path(cfg.cptac.json_output) / "test.json"
    extract_bioptimus_embeddings(cfg, test_json, data = "test")
