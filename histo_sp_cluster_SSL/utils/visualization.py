import os
import csv
import math
import time
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt      

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

     


def plot_epoch_losses(losses_all, contrastive_losses, fn_losses, neighbor_losses, epoch, save_dir):
    """
    Plots average loss values per epoch.
    Each list should have one entry per epoch.
    """


    epochs = range(1, len(losses_all) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, losses_all, label="Total Loss", marker='o')
    plt.plot(epochs, contrastive_losses, label="Contrastive Loss", marker='x')
    plt.plot(epochs, fn_losses, label="False Negative Loss", marker='s')
    plt.plot(epochs, neighbor_losses, label="Neighbor Loss", marker='^')

    plt.xlabel("Batch")
    plt.ylabel("Average Loss")
    plt.title("Loss Breakdown per Batch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"epoch{epoch}_loss_plot.png")
    plt.savefig(save_path)
    plt.close("all") 

def plot_cluster_losses(losses_all, contrastive_losses, fn_losses, neighbor_losses, save_dir):
    """
    Plots average loss values per epoch.
    Each list should have one entry per epoch.
    """


    epochs = range(1, len(losses_all) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, losses_all, label="Total Loss", marker='o')
    plt.plot(epochs, contrastive_losses, label="Contrastive Loss", marker='x')
    plt.plot(epochs, fn_losses, label="False Negative Loss", marker='s')
    plt.plot(epochs, neighbor_losses, label="Neighbor Loss", marker='^')

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Loss Breakdown per epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_loss_plot.png")
    plt.savefig(save_path)
    plt.close("all")    


def plot_lr_schedule(lr_history, save_dir="."):
    """
    Plots the learning rate schedule over training steps.

    Args:
        lr_history: List of learning rate values recorded during training.
        save_dir: Directory to save the plot.
        run_name: Name of the training run for filename.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, label="Learning Rate")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lr_schedule.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Learning rate plot saved to {save_path}")           