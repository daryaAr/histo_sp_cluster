import os
import csv
import math
import time
import torch
import logging
import glob
import numpy as np
import matplotlib.pyplot as plt



# Learning Rate Scheduler (Cosine with Warmup) - Manual
def adjust_learning_rate(optimizer, epoch, base_lr, total_epochs):
    """
    MoCo v2: Linear warmup for first 10 epochs, then cosine decay.
    """
    warmup_epochs = 10  
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs  # Linear warmup
    else:
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))  # Cosine decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Apply LR to optimizer
    
    return lr  # Return LR for logging

def update_template_csv(csv_path, batch_size, temperature, avg_loss, train_time, num_epochs):
    """
    Update the CSV file with training details.
    """
    csv_headers = [
        "Batch Size", "Temperature", "Average Training Loss", 
        "Training Time", "Metric Type", "Number of Epochs"
    ]
    row_data = [
        batch_size, temperature, avg_loss, train_time, 
        "ContrastiveLoss", num_epochs
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(row_data)    

def save_checkpoint(epoch, model, optimizer, scaler, base_lr, step, checkpoint_path, best=False):
    """ Save model, optimizer, scaler, and learning rate state. """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'learning_rate': base_lr,
        'current_step': step # Save base LR
    }
    torch.save(checkpoint, checkpoint_path)
    #logger.info(f"Checkpoint saved: {checkpoint_path}")

    if best:
        torch.save(checkpoint, checkpoint_path)
        #logger.info(f"New best model saved: {best_path}")
    else:
        torch.save(checkpoint, checkpoint_path)  

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "superpixel_org_*.pth")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None  

def load_existing_losses(plot_save_dir):
    """Load previous loss values if they exist."""
    loss_file = os.path.join(plot_save_dir, f"training_loss.txt")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            losses = [float(line.strip()) for line in f.readlines()]
        return losses
    return []

def save_losses(losses, plot_save_dir):
    """Save the loss values to a file."""
    
    loss_file = os.path.join(plot_save_dir, "training_loss.txt")
    
    with open(loss_file, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n") 

def save_losses_cluster(losses_all, contrastive_losses, fn_losses, neighbor_losses, plot_save_dir):
    """Save the loss values to a file."""
    
    loss_file = os.path.join(plot_save_dir, "training_loss.txt")
    contrastive_loss_file = os.path.join(plot_save_dir, "training_contrastive_loss.txt")
    fn_loss_file = os.path.join(plot_save_dir, "training_fn_loss.txt")
    neighbor_loss_file = os.path.join(plot_save_dir, "training_neighbor_loss.txt")
    
    with open(loss_file, "w") as f:
        for loss in losses_all:
            f.write(f"{loss}\n")
    with open(contrastive_loss_file, "w") as f:
        for loss in contrastive_losses:
            f.write(f"{loss}\n")
    with open(fn_loss_file, "w") as f:
        for loss in fn_losses:
            f.write(f"{loss}\n")
    with open(neighbor_loss_file, "w") as f:
        for loss in neighbor_losses:
            f.write(f"{loss}\n")                          

def log_training_run(cfg, training_loss, run_name):
    """
    Logs the training configuration and final training loss into a CSV file.

    Args:
        cfg: configuration object (from Hydra/OmegaConf or similar)
        training_loss: final training loss (float)
        run_name: name of the current run (str)
        csv_path: path to the CSV file
    """
    header = [
        "run_name",
        "base_encoder",
        "moco_type",
        "output_dim",
        "queue_size",
        "num_clusters",
        "momentum",
        "temperature",
        "init_queue_type",
        "batch_size",
        "no. epochs",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "alpha (fn)",
        "beta (fn)",
        "lambda_bml",
        "cluster_type",
        "training_loss"
    ]

    row = {
        "run_name": run_name,
        "base_encoder": cfg.model.base_encoder,
        "moco_type": cfg.model.moco_type,
        "output_dim": cfg.model.output_dim,
        "queue_size": cfg.model.queue_size,
        "num_clusters": getattr(cfg.model, "num_clusters", None),
        "momentum": cfg.model.momentum,
        "temperature": cfg.model.temperature,
        "init_queue_type": getattr(cfg.model, "init_queue_type", None),
        "batch_size": cfg.training.batch_size,
        "no. epochs": cfg.training.epochs,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "optimizer": cfg.training.optimizer,
        "alpha (fn)": getattr(cfg.training, "alpha", None),
        "beta (fn)": getattr(cfg.training, "beta", None),
        "lambda_bml": getattr(cfg.training, "lambda_bml", None),
        "cluster_type": getattr(cfg.cluster, "cluster_type", None),
        "training_loss": training_loss,
    }
    csv_path = cfg.paths.train_csv_path
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)            