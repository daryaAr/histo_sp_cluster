import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
from histo_MIL.config import load_yaml_config
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence




class GlobalEmbeddingMILWSIDataset(Dataset):
    def __init__(self, embeddings_path, tile_paths_path, metadata_path, label_mapping):
        self.embeddings = torch.load(embeddings_path, weights_only=True)  # shape [N_tiles, D]
        tile_paths = torch.load(tile_paths_path)       # list of str
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
        tile_embeddings = self.embeddings[item["tile_indices"]]  # [N_tiles, D]
        label = torch.tensor(item["label"], dtype=torch.long)
        return tile_embeddings, label


class PackedEmbeddingMILWSIDataset(Dataset):
    def __init__(self, embedding_file, metadata_path, label_mapping):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tiles = self.embeddings[item["tile_indices"]]  # [N_tiles, D]
        label = torch.tensor(item["label"], dtype=torch.long)
        return tiles, label

def collate_fn_ragged(batch):
    bags, labels = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True)  # [B, max_len, D]
    lengths = torch.tensor([x.size(0) for x in bags])
    labels = torch.stack(labels)
    return padded_bags, lengths, labels              


def get_dataset(cfg, metadata_path, embeddings_path):
    label_mapping = cfg.mil.label_mapping

    if cfg.embeddings.type == "bioptimus":
        data_type = cfg.mil.mode
        tile_paths = Path(cfg.cptac.embeddings_result) / cfg.embeddings.type / f"{data_type}_tile_paths.pt"
        return GlobalEmbeddingMILWSIDataset(
            embeddings_path=embeddings_path,
            tile_paths_path=tile_paths,
            metadata_path=metadata_path,
            label_mapping=label_mapping
        )
    else:
        return PackedEmbeddingMILWSIDataset(
            embedding_file=embeddings_path,
            metadata_path=metadata_path,
            label_mapping=label_mapping
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
 #   test_dataloaders()