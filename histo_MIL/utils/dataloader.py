import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
from histo_MIL.config import load_yaml_config
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

def analyze_wsi_tile_counts(dataset):
    bag_sizes = [item["tile_indices"].__len__() for item in dataset.data]
    bag_sizes = np.array(bag_sizes)

    q1 = np.percentile(bag_sizes, 25)
    q3 = np.percentile(bag_sizes, 75)
    iqr = q3 - q1

    print(f"WSI tile count statistics:")
    print(f"Min: {bag_sizes.min()}")
    print(f"Max: {bag_sizes.max()}")
    print(f"Median: {np.median(bag_sizes)}")
    print(f"IQR: {iqr} (Q1={q1}, Q3={q3})")

    # Optional: plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(bag_sizes, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(q1, color='orange', linestyle='--', label='Q1')
    plt.axvline(q3, color='green', linestyle='--', label='Q3')
    plt.title("Number of Tiles per WSI")
    plt.xlabel("Number of Tiles")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()




class GlobalEmbeddingMILWSIDataset(Dataset):
    def __init__(self, embeddings_path, tile_paths_path, metadata_path, label_mapping, max_tiles=None):
        self.embeddings = torch.load(embeddings_path, weights_only=True)
        self.max_tiles = max_tiles  # shape [N_tiles, D]
        tile_paths = torch.load(tile_paths_path)       
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Map: tile_path -> (wsi_id, label)
        path_to_info = {}
        for entry in metadata:
            wsi_dir = entry["wsi_dir"]
            wsi_id = Path(wsi_dir).name
            label = label_mapping[entry["label"]]
            for tile_path in entry["tiles_files"]:
                path_to_info[tile_path] = (wsi_id, label)

        # Group indices of tiles per WSI
        grouped = defaultdict(list)
        wsi_labels = {}
        for idx, path in enumerate(tile_paths):
            if path in path_to_info:
                wsi_id, label = path_to_info[path]
                grouped[wsi_id].append(idx)
                wsi_labels[wsi_id] = label  # only one label per WSI

        # Final data entries
        self.data = []
        for wsi_id, indices in grouped.items():
           
            self.data.append({
                    "wsi_id": wsi_id,
                    "tile_indices": indices,
                    "label": wsi_labels[wsi_id]
                })    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tile_indices = item["tile_indices"]

        if self.max_tiles is not None:
            if len(tile_indices) > self.max_tiles:
                tile_indices = random.sample(tile_indices, self.max_tiles)

        tile_embeddings = self.embeddings[tile_indices]  # [max_tiles, D]
        label = torch.tensor(item["label"], dtype=torch.long)
        return tile_embeddings, label


class PackedEmbeddingMILWSIDataset(Dataset):
    def __init__(self, embedding_file, metadata_path, label_mapping, max_tiles = None):
        result = torch.load(embedding_file)
        self.embeddings = result["embeddings"]
        self.paths = result["paths"]
        self.max_tiles = max_tiles
        self.tile_info = result.get("tile_info", None)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        wsi_labels = {e["wsi_id"]: label_mapping[e["label"]] for e in metadata}
        grouped = defaultdict(list)
        for idx, path in enumerate(self.paths):
            wsi_id = Path(path).parts[-3]
            grouped[wsi_id].append(idx)

        self.data = [
            {"wsi_id": wsi_id, "tile_indices": indices, "label": wsi_labels[wsi_id]}
            for wsi_id, indices in grouped.items()
            #if wsi_id in wsi_labels and 500 <= len(indices) <= 3000
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tile_indices = item["tile_indices"]
        label = torch.tensor(item["label"], dtype=torch.long)

        # Randomly sample up to max_tiles (e.g., 500)
         
        
        if self.max_tiles is not None:
            if len(tile_indices) > self.max_tiles:
                tile_indices = random.sample(tile_indices, self.max_tiles)

        tiles = self.embeddings[tile_indices]  # [num_sampled_tiles, D]
        return tiles, label
"""
class GlobalEmbeddingMILWSIDataset(Dataset):
    def __init__(self, embeddings_path, tile_paths_path, metadata_path, label_mapping, project_dim=None):
        self.embeddings = torch.load(embeddings_path, weights_only=True)
        tile_paths = torch.load(tile_paths_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        path_to_info = {}
        for entry in metadata:
            wsi_dir = entry["wsi_dir"]
            wsi_id = Path(wsi_dir).name
            label = label_mapping[entry["label"]]
            for tile_path in entry["tiles_files"]:
                path_to_info[tile_path] = (wsi_id, label)

        grouped = defaultdict(list)
        wsi_labels = {}
        for idx, path in enumerate(tile_paths):
            if path in path_to_info:
                wsi_id, label = path_to_info[path]
                grouped[wsi_id].append(idx)
                wsi_labels[wsi_id] = label

        self.data = [{"wsi_id": wsi_id, "tile_indices": indices, "label": wsi_labels[wsi_id]} for wsi_id, indices in grouped.items()]
        
        self.projection = None
        if project_dim:
            original_dim = self.embeddings.shape[1]
            self.projection = nn.Linear(original_dim, project_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tiles = self.embeddings[item["tile_indices"]]  # [N_tiles, D]
        if self.projection:
            tiles = self.projection(tiles).detach()
        label = torch.tensor(item["label"], dtype=torch.long)
        return tiles, label
    

class PackedEmbeddingMILWSIDataset(Dataset):
    def __init__(self, embedding_file, metadata_path, label_mapping, project_dim=None):
        result = torch.load(embedding_file)
        self.embeddings = result["embeddings"]
        self.paths = result["paths"]
        self.tile_info = result.get("tile_info", None)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        wsi_labels = {e["wsi_id"]: label_mapping[e["label"]] for e in metadata}
        grouped = defaultdict(list)
        for idx, path in enumerate(self.paths):
            wsi_id = Path(path).parts[-3]
            grouped[wsi_id].append(idx)

        self.data = [
            {"wsi_id": wsi_id, "tile_indices": indices, "label": wsi_labels[wsi_id]}
            for wsi_id, indices in grouped.items() if wsi_id in wsi_labels
        ]

        self.projection = None
        if project_dim:
            original_dim = self.embeddings.shape[1]
            self.projection = nn.Linear(original_dim, project_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tiles = self.embeddings[item["tile_indices"]]  # [N_tiles, D]
        if self.projection:
            tiles = self.projection(tiles).detach()
        label = torch.tensor(item["label"], dtype=torch.long)
        return tiles, label

"""

def collate_fn_ragged(batch):
    bags, labels = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True)  # [B, max_len, D]
    lengths = torch.tensor([x.size(0) for x in bags])
    labels = torch.stack(labels)
    return padded_bags, lengths, labels              


def get_dataset(cfg, metadata_path, embeddings_path):
    label_mapping = cfg.mil.label_mapping
    if cfg.embeddings.projection:
        project_dim = cfg.embeddings.projection_dim
    else:
         project_dim = None  
    max_tiles = int(cfg.training.bag) if cfg.training.bag is not None else None      

    if cfg.embeddings.type == "bioptimus":
        tile_paths = Path(cfg.cptac.embeddings_result) / cfg.embeddings.type / f"{cfg.mil.mode}_tile_paths.pt"
        return GlobalEmbeddingMILWSIDataset(
            embeddings_path=embeddings_path,
            tile_paths_path=tile_paths,
            metadata_path=metadata_path,
            label_mapping=label_mapping,
            max_tiles=max_tiles
        )
    else:
        return PackedEmbeddingMILWSIDataset(
            embedding_file=embeddings_path,
            metadata_path=metadata_path,
            label_mapping=label_mapping,
            max_tiles=max_tiles
        )


def get_train_val_loaders(cfg, metadata_path, embeddings_path, val_split: float = 0.2, seed: int = 42):
    dataset = get_dataset(cfg, metadata_path, embeddings_path)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, collate_fn=collate_fn_ragged)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, collate_fn=collate_fn_ragged)
    return train_loader, val_loader


def get_test_loader(cfg, metadata_path, embeddings_path): 
    dataset = get_dataset(cfg, metadata_path, embeddings_path)
    return DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                      num_workers=cfg.training.num_workers, collate_fn=collate_fn_ragged)

def test_dataloaders():
    cfg = load_yaml_config()

    print("\n--- Testing Train/Val Loaders ---")
    metadata_path = Path(cfg.cptac.json_output)/f"updated_{cfg.mil.data}.json"
    embeddings_path = Path(cfg.cptac.embeddings_result) / cfg.embeddings.type / f"{cfg.mil.data}.pt"
    train_loader, val_loader = get_train_val_loaders(cfg, metadata_path, embeddings_path)
    for i, (tiles, lengths, labels) in enumerate(train_loader):
        print(f"[Train] Batch {i}: {tiles.shape=}, {labels.shape=}")
        if i >= 1:
            break
    for i, (tiles, lengths, labels) in enumerate(val_loader):
        print(f"[Val] Batch {i}: {tiles.shape=}, {labels.shape=}")
        if i >= 1:
            break

    


#if __name__ == "__main__":
    #test_dataloaders()
        