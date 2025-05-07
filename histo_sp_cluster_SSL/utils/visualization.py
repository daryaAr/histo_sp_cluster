import os
import csv
import math
import time
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import umap


def plot_umap_with_queries(
    memory_bank_embeddings: np.ndarray,
    memory_bank_cluster_ids: np.ndarray,
    centroids: np.ndarray = None,
    queries: np.ndarray = None,
    query_cluster_ids: np.ndarray = None,
    save_path: str = None,
    annotate_query_ids: bool = False
):
    """
    Plots UMAP of memory bank embeddings with optional overlay of query embeddings,
    colored by cluster assignment. Optionally annotate queries with their cluster ID.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    all_embeddings = memory_bank_embeddings

    if queries is not None:
        all_embeddings = np.concatenate([memory_bank_embeddings, queries], axis=0)

    embedding_2d = reducer.fit_transform(all_embeddings)
    mb_2d = embedding_2d[:len(memory_bank_embeddings)]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        mb_2d[:, 0],
        mb_2d[:, 1],
        c=memory_bank_cluster_ids,
        cmap="nipy_spectral",
        s=2,
        alpha=0.6,
        label="Memory Bank"
    )

    # Overlay query embeddings
    if queries is not None and query_cluster_ids is not None:
        q_2d = embedding_2d[len(memory_bank_embeddings):]
        plt.scatter(
            q_2d[:, 0],
            q_2d[:, 1],
            c=query_cluster_ids,
            cmap="nipy_spectral",
            s=35,
            marker="o",
            edgecolor="black",
            linewidth=0.7,
            label="Queries"
        )

        if annotate_query_ids:
            for i, cid in enumerate(query_cluster_ids):
                plt.annotate(
                    str(cid),
                    (q_2d[i, 0], q_2d[i, 1]),
                    fontsize=6,
                    color='black',
                    alpha=0.8,
                    ha='center',
                    va='center'
                )

    # Optional centroid overlay
    if centroids is not None:
        centroid_2d = reducer.transform(centroids)
        plt.scatter(
            centroid_2d[:, 0],
            centroid_2d[:, 1],
            c="black",
            s=60,
            marker="X",
            label="Centroids"
        )

    plt.title("UMAP: Memory Bank + Queries (colored by Cluster ID)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()          

def plot_loss_curve(losses, save_path, title="Training Loss Curve"):
    """
    Plots and saves the training loss curve.

    Args:
        losses (list[float]): List of loss values per epoch.
        save_path (str): Full path to save the loss curve image.
        title (str): Title of the plot.
    """
    if not losses:
        raise ValueError("Loss list is empty. Cannot plot.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  

def plot_live_losses(x_labels, losses_all, contrastive_losses, fn_losses, neighbor_losses, save_dir,epoch, batch_idx):
    plt.figure(figsize=(10, 6))
    x_range = range(len(x_labels))
    plt.plot(x_range, losses_all, label="Total Loss", marker='o')
    plt.plot(x_range, contrastive_losses, label="Contrastive Loss", marker='x')
    plt.plot(x_range, fn_losses, label="False Negative Loss", marker='s')
    plt.plot(x_range, neighbor_losses, label="Neighbor Loss", marker='^')

    plt.xticks(ticks=x_range, labels=x_labels, rotation=45, fontsize=7)
    plt.xlabel("Epoch_Batch")
    plt.ylabel("Loss")
    plt.title(f"Loss Breakdown - Epoch {epoch}, Batch {batch_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"loss_{epoch}_{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()          