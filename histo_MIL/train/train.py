import os
import logging
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from histo_MIL.config import load_yaml_config
from histo_MIL.utils.dataloader import get_train_val_loaders
from histo_MIL.train.train_setup import train_mil
from histo_MIL.utils.logger import logger
from histo_MIL.utils.utils import build_run_name, set_embedding_dim
from histo_MIL.model.MIL_Attention import AttentionMILPL
from pathlib import Path


if __name__ == "__main__":
    cfg = load_yaml_config()

    # === Ask user for inputs ===
    def safe_input(prompt, default, cast_func=str):
        user_input = input(f"{prompt} (default {default}): ").strip()
        return cast_func(user_input) if user_input else default
    

    cfg.embeddings.type = safe_input(
        "Enter embedding type (resnet50 / bioptimus / mocov2 / superpixel_cluster / superpixel / superpixel_cluster_wofn / superpixel_cluster_wofn_withproj / superpixel_cluster_wofn_merged)",
        cfg.embeddings.get("type", "bioptimus"),
        str
    )
    projection = input(f"Using projection in embeddings (yes/no, default {'yes' if cfg.embeddings.projection else 'no'}): ").strip().lower()
    if projection:
        cfg.embeddings.projection = True if projection == "yes" else False
    cfg.training.device = safe_input("Enter device (cuda:0 / cuda:1)", cfg.training.get("device", "cuda:0"), str)
    cfg.mil.mode = safe_input("Enter mode of the run", cfg.mil.get("mode", "train"), str)
    cfg.training.batch_size = safe_input("Enter batch size", cfg.training.batch_size, int)
    cfg.training.epochs = safe_input("Enter number of epochs", cfg.training.epochs, int)
    cfg.training.learning_rate = safe_input("Enter learning rate", cfg.training.learning_rate, float)
    cfg.model.dropout = safe_input("Enter dropout", cfg.model.dropout, float)
    cfg.training.num_workers = safe_input("Enter number of workers", cfg.training.num_workers, int)

    set_embedding_dim(cfg)
    run_name = build_run_name(cfg)

    # === Setup output directories ===
    cfg.mil_paths.save_model_dir = os.path.join(cfg.mil_paths.model_save_dir, cfg.mil.mode, run_name)
    cfg.mil_paths.checkpoint_dir = os.path.join(cfg.mil_paths.checkpoint_dir, run_name)
    cfg.mil_paths.confusion_dir = os.path.join(cfg.mil_paths.output_base, "confusion matrix", run_name)
    cfg.mil_paths.loss_log_dir = os.path.join(cfg.mil_paths.output_base, "loss log", run_name)
    cfg.mil_paths.loss_plot_dir = os.path.join(cfg.mil_paths.output_base, "loss plot", run_name)
    cfg.mil_paths.metric_plot_dir = os.path.join(cfg.mil_paths.output_base, "metric plot", run_name)
    cfg.mil_paths.tensorboard_dir = os.path.join(cfg.mil_paths.output_base, "tensorboard", run_name)
    
    for p in [
        cfg.mil_paths.save_model_dir,
        cfg.mil_paths.checkpoint_dir,
        cfg.mil_paths.confusion_dir,
        cfg.mil_paths.loss_log_dir,
        cfg.mil_paths.tensorboard_dir,
        cfg.mil_paths.loss_plot_dir,
        cfg.mil_paths.metric_plot_dir
    ]:
        os.makedirs(p, exist_ok=True)

    # === Logger ===
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {cfg.training.device}")
    logger.info(f"Run name: {run_name}")

    # === Prepare model ===
    model = AttentionMILPL(
        input_dim=cfg.embeddings.dim,
        hidden_dim=cfg.model.hidden_dim,
        attention_dim=cfg.model.attention_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        attention_branches=cfg.model.attention_branches
    )

    train_loader, val_loader = get_train_val_loaders(
        cfg,
        Path(cfg.mil_paths.data_json) / "updated_train.json",
        Path(cfg.cptac.embeddings_result) / cfg.embeddings.type / "train.pt" #"superpixel_cluster_wofn" / "train.pt"
    )
    writer = SummaryWriter(log_dir=cfg.mil_paths.tensorboard_dir)

    # === Start training ===
    train_mil(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader, writer=writer, run_name=run_name)