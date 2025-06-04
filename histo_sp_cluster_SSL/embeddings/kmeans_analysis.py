import os
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from histo_sp_cluster_SSL.config import load_yaml_config


def analyze_cluster_majority_agreement(data):
    spx_to_tiles = defaultdict(list)
    for entry in data:
        spx_to_tiles[entry["superpixel_id"]].append(entry)

    per_superpixel_percentages = []

    for tiles in spx_to_tiles.values():
        
        primary_clusters = [t["primary_cluster"] for t in tiles]
        cluster_counts = Counter(primary_clusters)
        most_common_cluster, count = cluster_counts.most_common(1)[0]
        percentage = count / len(tiles)
        per_superpixel_percentages.append(percentage)

    average_percentage = sum(per_superpixel_percentages) / len(per_superpixel_percentages)


    return average_percentage, per_superpixel_percentages


def compute_negative_tile_overlap_percentage(data):
    # Group all tiles by superpixel and by cluster
    spx_to_tiles = defaultdict(list)
    cluster_to_tiles = defaultdict(list)

    for entry in data:
        spx_to_tiles[entry["superpixel_id"]].append(entry)
        cluster_to_tiles[entry["primary_cluster"]].append(entry)

    per_tile_percentages = []

    for tile in data:
        spx_id = tile["superpixel_id"]
        c2 = tile["second_cluster"]
        c3 = tile["third_cluster"]

        # Get all negative tiles (primary cluster == c2 or c3)
        negative_tiles = cluster_to_tiles[c2] + cluster_to_tiles[c3]

        if len(negative_tiles) == 0:
            continue  # Avoid divide-by-zero

        # Count how many of them are in the same superpixel
        same_spx_negatives = [t for t in negative_tiles if t["superpixel_id"] == spx_id]

        percentage = (len(same_spx_negatives) / len(negative_tiles)) * 100
        per_tile_percentages.append(percentage)

    global_average_percentage = sum(per_tile_percentages) / len(per_tile_percentages)
    return global_average_percentage, per_tile_percentages


def main():
    cfg = load_yaml_config()

    per_tile_json_path = os.path.join(
        cfg.paths.output_base,
        "bioptimus_embeddings", "cluster_json", "json_bioptimus_cluster50.json"
    )
    save_dir = os.path.join(cfg.paths.output_base, "bioptimus_embeddings", "cluster_json")

    # === Load data ===
    with open(per_tile_json_path, "r") as f:
        data = json.load(f)

    # === Analyze and visualize cluster distribution ===
    print("=== Analyzing cluster statistics ===")
    avg, per_superpixel_percentages = analyze_cluster_majority_agreement(data)
    print(f"Average percentage of agreement with dominant cluster: {avg:.4f}")

    # === Bar plot: Number of tiles per primary cluster ===
    print("=== Plotting tile count per primary cluster ===")
    cluster_ids = [entry["primary_cluster"] for entry in data]
    counts = Counter(cluster_ids)

    avg_neg_overlap, per_tile_scores = compute_negative_tile_overlap_percentage(data)

    print(f"📊 Average % of negative tiles in same superpixel: {avg_neg_overlap:.2f}%")
    print(f"Total tiles analyzed: {len(per_tile_scores)}")

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
    barplot_path = os.path.join(save_dir, "primary_cluster_tile_counts50.png")
    plt.savefig(barplot_path)
    plt.close()
    print(f"Bar plot saved: {barplot_path}")


if __name__ == "__main__":
    main() 