import pandas as pd
import json
import torch
import random
import umap
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from histo_MIL.config import load_yaml_config
from histo_MIL.utils.logger import logger
from histo_MIL.utils.utils import build_run_name, set_embedding_dim
import os
import numpy as np
import pickle

class EmbeddingVisualizer:
    def __init__(self, embedding_path, neighbor_json_path, output_path, model_type="default"):
        self.embedding_path = Path(embedding_path)
        self.neighbor_json_path = Path(neighbor_json_path)
        self.model_type = model_type
        self.output_path = output_path
        
        
        self.embeddings = None
        self.paths = None
        self.tile_info = None
        self.tile_name_to_embedding = {}
        self.wsi_to_entries = defaultdict(list)

    def prepare_subset_tile_list(self, num_wsis=30, tiles_per_wsi=6, seed=42):
        random.seed(seed)

        # Load neighbor JSON
        with open(self.neighbor_json_path, "r") as f:
            neighbor_data = json.load(f)

        # Organize entries by WSI
        wsi_to_entries = defaultdict(list)
        for entry in neighbor_data:
            wsi_to_entries[entry["wsi_id"]].append(entry)

        selected_wsis = random.sample(list(wsi_to_entries.keys()), min(num_wsis, len(wsi_to_entries)))
        selected_tile_names = set()

        for wsi_id in selected_wsis:
            entries = wsi_to_entries[wsi_id]
            valid_entries = [e for e in entries if isinstance(e["center_tile"], str)]
            if len(valid_entries) < tiles_per_wsi:
                continue

            chosen_entries = random.sample(valid_entries, tiles_per_wsi)

            for entry in chosen_entries:
                center = Path(entry["center_tile"]).stem
                selected_tile_names.add(center)
                for neighbor_path in entry["neighbors"]:
                    neighbor = Path(neighbor_path).stem
                    selected_tile_names.add(neighbor)

        # Save tile names to JSON
        
        output_path = Path(self.output_path)/ "subset_tile_names.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(sorted(list(selected_tile_names)), f, indent=2)

        print(f"Saved {len(selected_tile_names)} tile names to {output_path}")    

    def load_data(self, subset_tile_list_path=None):
        subset_tile_names = None
        if subset_tile_list_path:
            with open(subset_tile_list_path, "r") as f:
                subset_tile_names = set(json.load(f))
            print(f"Filtering embeddings to {len(subset_tile_names)} selected tiles")

        if self.model_type == "bioptimus":
            # Bioptimus format: separate .pt files for embeddings and paths
            embedding_tensor_path = self.embedding_path
            tile_path_file = self.embedding_path.parent / self.embedding_path.name.replace("train.pt", "train_tile_paths.pt")

            all_embeddings = torch.load(embedding_tensor_path)
            all_paths = torch.load(tile_path_file)
            self.tile_info = None

            if subset_tile_names:
                filtered = [
                    (path, emb) for path, emb in zip(all_paths, all_embeddings)
                    if Path(path).stem in subset_tile_names
                ]
                self.paths, emb_list = zip(*filtered)
                self.embeddings = torch.stack(emb_list)
            else:
                self.paths = all_paths
                self.embeddings = all_embeddings

            self.tile_name_to_embedding = {
                Path(path).stem: emb for path, emb in zip(self.paths, self.embeddings)
            }
            print(f"Loaded {len(self.embeddings)} Bioptimus embeddings")

        else:
            # MoCo format: dict with "embeddings", "paths", and optionally "tile_info"
            result = torch.load(self.embedding_path)
            all_embeddings = result["embeddings"]
            all_paths = result["paths"]
            self.tile_info = result.get("tile_info", None)

            if subset_tile_names:
                filtered = [
                    (path, emb) for path, emb in zip(all_paths, all_embeddings)
                    if Path(path).stem in subset_tile_names
                ]
                self.paths, emb_list = zip(*filtered)
                self.embeddings = torch.stack(emb_list)
            else:
                self.paths = all_paths
                self.embeddings = all_embeddings

            self.tile_name_to_embedding = {
                Path(path).stem: emb for path, emb in zip(self.paths, self.embeddings)
            }
            print(f"Loaded {len(self.embeddings)} embeddings for model_type={self.model_type}")

    def plot_multiple_wsi_embeddings(self, num_groups=6, seed=42, save_dir=None):
        random.seed(seed)

        with open(self.neighbor_json_path, "r") as f:
            neighbor_data = json.load(f)

        filtered_entries = [
            entry for entry in neighbor_data
            if Path(entry["center_tile"]).stem in self.tile_name_to_embedding
        ]

        wsi_to_entries = defaultdict(list)
        for entry in filtered_entries:
            wsi_to_entries[entry["wsi_id"]].append(entry)

        selected_wsis = random.sample(list(wsi_to_entries.keys()), k=4)
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()
        reducer = umap.UMAP(random_state=seed)
        palette = ["red", "blue", "green", "purple", "orange", "brown", "pink", "cyan"]

        for plot_idx, wsi_id in enumerate(selected_wsis):
            entries = wsi_to_entries[wsi_id]
            if len(entries) < num_groups:
                continue

            selected_entries = random.sample(entries, num_groups)
            all_tile_names = []
            all_colors = []
            center_indices = []
            group_labels = []
            distance_per_group = []

            for i, entry in enumerate(selected_entries):
                center = Path(entry["center_tile"]).stem
                neighbors = [Path(n).stem for n in entry["neighbors"]]
                group_tile_names = [center] + [n for n in neighbors if n in self.tile_name_to_embedding]
                group_embs = [self.tile_name_to_embedding[t] for t in group_tile_names]

                center_idx = len(all_tile_names)
                center_indices.append(center_idx)
                all_tile_names.extend(group_tile_names)
                all_colors.extend([palette[i % len(palette)]] * len(group_tile_names))

                group_tensor = torch.stack(group_embs)
                var = group_tensor.var().item()
                mean = group_tensor.mean().item()
                group_labels.append(f"G{i+1} (var: {var:.2f})")

            tile_embeddings = torch.stack([self.tile_name_to_embedding[t] for t in all_tile_names]).numpy()
            emb_2d = reducer.fit_transform(tile_embeddings)

            ax = axs[plot_idx]
            for i in range(num_groups):
                color = palette[i % len(palette)]
                group_idxs = [j for j in range(len(all_colors)) if all_colors[j] == color]
                ax.scatter(emb_2d[group_idxs, 0], emb_2d[group_idxs, 1], color=color, alpha=0.8, label=group_labels[i])

            for i, center_idx in enumerate(center_indices):
                color = palette[i % len(palette)]
                ax.scatter(emb_2d[center_idx, 0], emb_2d[center_idx, 1], marker='*', color=color, edgecolors='k', s=200)

                group_idxs = [j for j in range(len(all_colors)) if all_colors[j] == color]
                neighbor_coords = [emb_2d[j] for j in group_idxs if j != center_idx]
                dist = np.mean([np.linalg.norm(emb_2d[center_idx] - p) for p in neighbor_coords])
                distance_per_group.append(dist)

            avg_wsi_dist = np.mean(distance_per_group) if distance_per_group else 0.0
            ax.set_title(f"WSI: {wsi_id}\nAvg center-neighbor dist: {avg_wsi_dist:.2f}")
            ax.legend()

        plt.tight_layout()
        if save_dir:
            save_path = Path(save_dir) / f"multi_umap_{self.model_type}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Saved to {save_path}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    cfg = load_yaml_config()

    # === Ask user for inputs ===
    def safe_input(prompt, default, cast_func=str):
        user_input = input(f"{prompt} (default {default}): ").strip()
        return cast_func(user_input) if user_input else default
    

    cfg.embeddings.type = safe_input(
        "Enter embedding type (resnet50 / bioptimus / mocov2 / superpixel_cluster / superpixel / superpixel_cluster_wofn / superpixel_cluster_wofn_withproj)",
        cfg.embeddings.get("type", "bioptimus"),
        str
    )
    set_embedding_dim(cfg)
    run_name = build_run_name(cfg)
    cfg.mil_paths.emb_plot_dir = os.path.join(cfg.mil_paths.output_base, "embeddings plot", run_name)
    cfg.mil_paths.selected_wsi = os.path.join(cfg.mil_paths.output_base, "selected_wsi")
    os.makedirs(cfg.mil_paths.emb_plot_dir, exist_ok=True)
    os.makedirs(cfg.mil_paths.selected_wsi, exist_ok=True)

    visualizer = EmbeddingVisualizer(
        embedding_path=Path(cfg.cptac.embeddings_result) / cfg.embeddings.type / "train.pt",
        neighbor_json_path=cfg.cptac.neighbors_json,
        output_path=cfg.mil_paths.selected_wsi,
        model_type=cfg.embeddings.type     
    )
    if cfg.visual.select:
        visualizer.prepare_subset_tile_list()

    visualizer.load_data(subset_tile_list_path=Path(cfg.mil_paths.selected_wsi)/ "subset_tile_names.json")
    visualizer.plot_multiple_wsi_embeddings(save_dir=cfg.mil_paths.emb_plot_dir)