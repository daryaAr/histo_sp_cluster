import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from histo_sp_cluster_SSL.config import load_yaml_config
from collections import Counter, defaultdict
from sklearn.decomposition import IncrementalPCA
from umap import UMAP



class EmbeddingDataset(Dataset):
    def __init__(self, emb_tensor_path):
        self.embeddings = torch.load(emb_tensor_path, map_location="cpu")  # shape (N, D)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx]


def cosine_kmeans(embeddings, num_clusters, num_iters=50, batch_size=8192, device="cuda"):
    dataloader = DataLoader(EmbeddingDataset(embeddings), batch_size=batch_size, shuffle=True)
    dim = next(iter(dataloader)).shape[1]

    # Initialize centroids with the first batch (already normalized)
    centroids = next(iter(dataloader)).to(device)
    centroids = centroids[:num_clusters]
    centroids = torch.nn.functional.normalize(centroids, dim=1)

    for epoch in range(num_iters):
        print(f"Iteration {epoch + 1}/{num_iters}")
        # To accumulate updates
        cluster_sums = torch.zeros_like(centroids)
        cluster_counts = torch.zeros(num_clusters, device=device)

        for batch in tqdm(dataloader, desc="Assigning & Updating"):
            batch = batch.to(device)
            batch = torch.nn.functional.normalize(batch, dim=1)

            sims = torch.matmul(batch, centroids.T)  # cosine similarity
            assignments = torch.argmax(sims, dim=1)

            for i in range(num_clusters):
                selected = batch[assignments == i]
                if selected.shape[0] > 0:
                    cluster_sums[i] += selected.sum(dim=0)
                    cluster_counts[i] += selected.shape[0]

        # Update centroids
        for i in range(num_clusters):
            if cluster_counts[i] > 0:
                centroids[i] = cluster_sums[i] / cluster_counts[i]
        centroids = torch.nn.functional.normalize(centroids, dim=1)

    return centroids


def assign_top3(embeddings, centroids, batch_size=8192, device="cuda"):
    dataloader = DataLoader(EmbeddingDataset(embeddings), batch_size=batch_size, shuffle=False)
    all_top3 = []

    centroids = centroids.to(device)

    for batch in tqdm(dataloader, desc="Assigning top 3 centroids"):
        batch = batch.to(device)
        batch = torch.nn.functional.normalize(batch, dim=1)

        sims = torch.matmul(batch, centroids.T)
        top3 = torch.topk(sims, k=3, dim=1).indices.cpu().tolist()
        all_top3.extend(top3)

    return all_top3

def analyze_cluster_distribution(per_tile_json_path, save_dir):
    """
    Analyzes how consistent the clusters are across tiles within each superpixel.

    Args:
        per_tile_json_path (str): Path to the per-tile JSON containing cluster assignments.
        save_dir (str): Path where to save plots.

    Returns:
        dict: Statistics with percentages of tiles whose primary cluster is also a secondary/tertiary of others.
    """
    with open(per_tile_json_path, "r") as f:
        data = json.load(f)

    # Group tiles by superpixel
    spx_to_tiles = defaultdict(list)
    for entry in data:
        spx_to_tiles[entry["superpixel_id"]].append(entry)

    total_tiles = 0
    primary_match_count = 0
    second_match_count = 0
    third_match_count = 0

    for spx_id, tiles in spx_to_tiles.items():
        primary_clusters = [t["primary_cluster"] for t in tiles]
        second_clusters = [t["second_cluster"] for t in tiles]
        third_clusters = [t["third_cluster"] for t in tiles]

        total_tiles += len(tiles)

        # For each tile in superpixel, check how many other tiles contain this tile's primary cluster in their 2nd/3rd
        for i, pc in enumerate(primary_clusters):
            others_second = second_clusters[:i] + second_clusters[i+1:]
            others_third = third_clusters[:i] + third_clusters[i+1:]

            if pc in primary_clusters[:i] + primary_clusters[i+1:]:
                primary_match_count += 1
            elif pc in others_second:
                second_match_count += 1
            elif pc in others_third:
                third_match_count += 1

    # === Summary stats ===
    stats = {
        "same_superpixel_primary_match": primary_match_count / total_tiles,
        "primary_in_second_of_others": second_match_count / total_tiles,
        "primary_in_third_of_others": third_match_count / total_tiles,
    }

    # === Optional: plot as pie chart ===
    plt.figure(figsize=(6, 6))
    labels = ["Primary", "In Second", "In Third", "Unmatched"]
    values = [
        primary_match_count,
        second_match_count,
        third_match_count,
        total_tiles - (primary_match_count + second_match_count + third_match_count)
    ]
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Tile Cluster Match Within Superpixels")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "superpixel_cluster_consistency_pie50.png")
    plt.savefig(save_path)
    plt.close()

    return stats

def stream_umap_plot(emb_path, top3_indices, save_path, batch_size=50000):
    print("Running streaming UMAP with incremental PCA...")

    # === Step 1: Stream embeddings from disk in batches
    embeddings = torch.load(emb_path, map_location="cpu")
    num_embeddings = embeddings.shape[0]

    # === Step 2: Use Incremental PCA for dimensionality reduction
    pca = IncrementalPCA(n_components=50, batch_size=batch_size)
    for i in range(0, num_embeddings, batch_size):
        batch = embeddings[i:i+batch_size].numpy()
        pca.partial_fit(batch)

    reduced_batches = []
    for i in range(0, num_embeddings, batch_size):
        batch = embeddings[i:i+batch_size].numpy()
        reduced = pca.transform(batch)
        reduced_batches.append(reduced)

    reduced_embeddings = np.vstack(reduced_batches)

    # === Step 3: Apply UMAP on reduced data
    reducer = UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(reduced_embeddings)

    # === Step 4: Plot UMAP
    labels = [c[0] for c in top3_indices]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=2)
    plt.colorbar(scatter)
    plt.title("Bioptimus Cluster UMAP")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP saved to: {save_path}")


def main():
    cfg = load_yaml_config()

    emb_path = os.path.join(cfg.paths.output_base, "bioptimus_embeddings", "bioptimus_embeddings.pt")
    json_path = cfg.paths.json_path
    save_json_dir = os.path.join(cfg.paths.output_base, "bioptimus_embeddings", "cluster_json")
    umap_dir = os.path.join(cfg.paths.output_base, "figures", "bioptimus_cluster_umap")

    os.makedirs(save_json_dir, exist_ok=True)
    os.makedirs(umap_dir, exist_ok=True)

    print("=== Running cosine-based k-means clustering ===")
    centroids = cosine_kmeans(emb_path, num_clusters=cfg.model.num_clusters, num_iters=50, device=cfg.training.device)

    print("=== Assigning top-3 centroids for each tile ===")
    top3_indices = assign_top3(emb_path, centroids, device=cfg.training.device)

    print("=== Loading original tile mapping ===")
    with open(json_path, "r") as f:
        dataset_json = json.load(f)

    # Flatten all tile paths
    all_tile_paths = []
    tile_to_superpixel = []
    for entry in dataset_json:
        for tile_path in entry["tile_paths"]:
            all_tile_paths.append(tile_path)
            tile_to_superpixel.append(entry["superpixel_id"])

    assert len(top3_indices) == len(all_tile_paths), f"Expected {len(all_tile_paths)} embeddings, got {len(top3_indices)}"

    # Build new JSON entries: one per tile
    tile_json = []
    for tile_path, spx_id, (c1, c2, c3) in zip(all_tile_paths, tile_to_superpixel, top3_indices):
        tile_json.append({
            "superpixel_id": spx_id,
            "tile_path": tile_path,
            "primary_cluster": int(c1),
            "second_cluster": int(c2),
            "third_cluster": int(c3)
        })

    json_save_path = os.path.join(save_json_dir, "json_bioptimus_cluster50.json")
    with open(json_save_path, "w") as f:
        json.dump(tile_json, f, indent=2)
    print(f"Saved enriched per-tile JSON: {json_save_path}")

    # === Analyze and visualize cluster distribution ===
    print("=== Analyzing cluster statistics ===")
    stats = analyze_cluster_distribution(json_save_path, save_json_dir)
    print("Superpixel cluster analysis:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")


    # === Bar plot: Number of tiles per primary cluster ===
    print("=== Plotting tile count per primary cluster ===")
    cluster_ids = [entry["primary_cluster"] for entry in tile_json]
    counts = Counter(cluster_ids)

    cluster_ids_sorted = sorted(counts.keys())
    values = [counts[cid] for cid in cluster_ids_sorted]

    plt.figure(figsize=(12, 6))
    plt.bar(cluster_ids_sorted, values)
    plt.xlabel("Primary Cluster ID")
    plt.ylabel("Number of Tiles")
    plt.title("Number of Tiles Assigned to Each Primary Cluster")
    plt.xticks(cluster_ids_sorted)
    plt.grid(True)
    plt.tight_layout()
    barplot_path = os.path.join(save_json_dir, "primary_cluster_tile_counts50.png")
    plt.savefig(barplot_path)
    plt.close()
    print(f"Bar plot saved: {barplot_path}")   

    #umap_path = os.path.join(umap_dir, "bioptimus_cluster_umap.png")
    #stream_umap_plot(emb_path, top3_indices, umap_path, batch_size=50000) 

    
    
    

if __name__ == "__main__":
    main()