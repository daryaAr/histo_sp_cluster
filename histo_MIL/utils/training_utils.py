import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import csv

def save_checkpoint(epoch, model, optimizer, scaler, base_lr, checkpoint_path, best=False):
    """ Save model, optimizer, scaler, and learning rate state. """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'learning_rate': base_lr,
    }
    torch.save(checkpoint, checkpoint_path)

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None

def save_mil_losses(losses, accuracies, f1_scores, epoch, plot_save_dir):
    """Save losses and metrics to separate text files."""
    os.makedirs(plot_save_dir, exist_ok=True)
    _save_single_metric(f"mil_loss_triggered_epoch{epoch+1}.txt", losses, plot_save_dir)
    _save_single_metric(f"mil_accuracy_triggered_epoch{epoch+1}.txt", accuracies, plot_save_dir)
    _save_single_metric(f"mil_f1_triggered_epoch{epoch+1}.txt", f1_scores, plot_save_dir)

def _save_single_metric(filename, values, save_dir):
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v:.6f}\n")

def load_mil_losses(plot_save_dir):
    """Load previously saved losses/metrics if they exist."""
    return {
        "loss": _load_single_metric(os.path.join(plot_save_dir, "mil_loss.txt")),
        "accuracy": _load_single_metric(os.path.join(plot_save_dir, "mil_accuracy.txt")),
        "f1": _load_single_metric(os.path.join(plot_save_dir, "mil_f1.txt")),
    }

def _load_single_metric(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [float(line.strip()) for line in f.readlines()]
    return []

def save_confusion_matrix(cm_tensor, class_names, save_path, epoch=None):
    """Save the confusion matrix as a heatmap image."""
    cm = cm_tensor.cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    title = f"Confusion Matrix"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def log_mil_training_run(cfg, run_name, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, epoch):
    """
    Logs the MIL training configuration and both training and validation metrics into a CSV file.

    Args:
        cfg: configuration object
        run_name: name of the current run (str)
        train_loss: final training loss (float)
        train_acc: final training accuracy (float)
        train_f1: final training F1 score (float)
        val_loss: final validation loss (float)
        val_acc: final validation accuracy (float)
        val_f1: final validation F1 score (float)
    """
    header = [
        "run_name",
        "trigerred epoch",
        "model_type",
        "input_dim",
        "hidden_dim",
        "attention_dim",
        "attention_heads",
        "num_classes",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "train_loss",
        "train_accuracy",
        "train_f1",
        "val_loss",
        "val_accuracy",
        "val_f1"
    ]

    row = {
        "run_name": run_name,
        "trigerred epoch": epoch+1,
        "model_type": cfg.embeddings.type,
        "input_dim": cfg.embeddings.dim,
        "hidden_dim": cfg.model.hidden_dim,
        "attention_dim": cfg.model.attention_dim,
        "attention_heads": cfg.model.attention_branches,
        "num_classes": cfg.model.num_classes,
        "dropout": cfg.model.dropout,
        "batch_size": cfg.training.batch_size,
        "epochs": cfg.training.epochs,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "optimizer": cfg.training.optimizer,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1
    }

    csv_path = cfg.mil_paths.train_csv_path
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)    

def plot_train_val_loss(train_losses, val_losses, ylabel, save_path):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (List[float]): List of training losses.
        val_losses (List[float]): List of validation losses.
        save_path (str or Path): Path to save the resulting plot (e.g., 'loss_curve.png').
    """
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label=f"Train {ylabel}", marker='o')
    plt.plot(epochs, val_losses, label=f"Validation {ylabel}", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Training & Validation {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()        