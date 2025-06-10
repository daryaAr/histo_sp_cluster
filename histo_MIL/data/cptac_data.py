import os
from torch.utils.tensorboard import SummaryWriter
import json
from histo_MIL.config import load_yaml_config
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def create_metadata(base_dir, label):
    base_dir = Path(base_dir)
    entries = []

    print(f"\n[INFO] Scanning folder: {base_dir} for label: {label}")
    wsi_folders = sorted([f for f in base_dir.iterdir() if f.is_dir()])

    for wsi_folder in tqdm(wsi_folders, desc=f"Processing {label}", unit="WSI"):

        tiles_dir = wsi_folder / "tiles"
        if not tiles_dir.exists() or not tiles_dir.is_dir():
            continue

        tiles_files = sorted([str(p) for p in tiles_dir.glob("*.png")])
        if not tiles_files:
            continue

        entries.append({
            "wsi_id": wsi_folder.name,
            "label": label,
            "wsi_dir": str(wsi_folder),
            "tiles_files": tiles_files
        })
    print(f"[DONE] Total WSIs processed for {label}: {len(entries)}\n")
    
    return entries

def save_json(entries, output_path):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(entries, f, indent=4)

def split_train_test(metadata, test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    test_ids = set(test_df["Slide_ID"].astype(str))

    test_entries = [entry for entry in metadata if entry["wsi_id"] in test_ids]
    train_entries = [entry for entry in metadata if entry["wsi_id"] not in test_ids]

    return train_entries, test_entries        


def main():
    cfg = load_yaml_config()

    output_dir = Path(cfg.cptac.json_output)
    luad_dir = Path(cfg.cptac.cptac_luad)
    lusc_dir = Path(cfg.cptac.cptac_lusc)
    test_csv = Path(cfg.cptac.cptac_test_csv)

    luad_entries = create_metadata(luad_dir, label="LUAD")
    lusc_entries = create_metadata(lusc_dir, label="LUSC")
    all_entries = luad_entries + lusc_entries

    save_json(luad_entries, output_dir / "cptac_luad.json")
    save_json(lusc_entries, output_dir / "cptac_lusc.json")
    save_json(all_entries, output_dir / "cptac.json")

    # Split into train/test
    train_entries, test_entries = split_train_test(all_entries, test_csv)
    save_json(train_entries, output_dir / "train.json")
    save_json(test_entries, output_dir / "test.json")

    # Split into train/test
    train_entries, test_entries = split_train_test(all_entries, test_csv)
    save_json(train_entries, output_dir / "train.json")
    save_json(test_entries, output_dir / "test.json")

    print(f"Metadata saved in {output_dir} (train/test + category-wise)")


if __name__ == "__main__":
    main()           