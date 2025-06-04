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
    loss_file = os.path.join(plot_save_dir, f"superpixel_org_loss_curve.txt")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            losses = [float(line.strip()) for line in f.readlines()]
        return losses
    return []

def save_losses(losses, plot_save_dir):
    """Save the loss values to a file."""
    os.makedirs(plot_save_dir, exist_ok=True)  # Ensure the directory exists
    
    loss_file = os.path.join(plot_save_dir, "superpixel_org_loss_curve.txt")
    
    with open(loss_file, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n") 