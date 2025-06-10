import os
import logging
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from histo_sp_cluster_SSL.config import load_yaml_config
from histo_sp_cluster_SSL.dataloader import get_dataloader
from histo_sp_cluster_SSL.train.train_setup import train_moco
from histo_sp_cluster_SSL.utils.logger import logger
from histo_sp_cluster_SSL.utils.utils import (
    get_model,
    get_loss_fn,
    build_run_name
)



if __name__ == "__main__":
    cfg = load_yaml_config()

    # === Ask user for dynamic terminal inputs (with defaults) ===
    def safe_input(prompt, default, cast_func=str):
        user_input = input(f"{prompt} (default {default}): ").strip()
        return cast_func(user_input) if user_input else default

    cfg.cluster.cluster_type = safe_input(
        "Enter cluster type (no_cluster/ cluster_resnet/ cluster_bioptimus/ cluster_uni)",
        cfg.cluster.get("cluster_type", "cluster_bioptimus"),
        str
    )
    cfg.model.moco_type = safe_input(
        "Enter MoCo type (moco_v2 / moco_superpixel / moco_superpixel_cluster / moco_superpixel_cluster_bioptimus)",
        cfg.model.get("moco_type", "moco_superpixel_cluster_bioptimus"),
        str
    )
    cfg.model.base_encoder = safe_input(
        "Enter base encoder (resnet18 / resnet34 / resnet50 / resnet101 / resnet152)",
        cfg.model.get("base_encoder", "resnet50"),
        str
    )
    cfg.model.init_queue_type = safe_input(
        "Enter queue initilizer type (random / awared_random)",
        cfg.model.get("init_queue_type", "awared_random"),
        str
    )
    cfg.model.momentum = safe_input("Enter momentum", cfg.model.momentum, int)
    cfg.model.num_clusters = safe_input("Enter number of queue clusters", cfg.model.num_clusters, int)
    cfg.model.temperature = safe_input("Enter temperture", cfg.model.temperature, int)
    cfg.loss.type = safe_input(
        "Enter loss type (cluster/ contrastive)",
        cfg.loss.get("type", "cluster"),
        str
    )
    neighbor = input(f"Using neighbors in Superpixel (yes/no, default {'yes' if cfg.training.use_neighbors else 'no'}): ").strip().lower()
    if neighbor:
        cfg.training.use_neighbors = True if neighbor == "yes" else False

    cfg.training.device = safe_input("Enter device (cuda:0 / cuda:1)", cfg.training.get("device", "cuda:0"), str)
    cfg.training.batch_size = safe_input("Enter batch size", cfg.training.batch_size, int)
    cfg.training.epochs = safe_input("Enter number of epochs", cfg.training.epochs, int)
    cfg.training.learning_rate = safe_input("Enter learning rate", cfg.training.learning_rate, float)
    cfg.training.alpha = safe_input("Enter alpha", cfg.training.alpha, float)
    cfg.training.beta = safe_input("Enter beta", cfg.training.beta, float)
    cfg.training.lambda_bml = safe_input("Enter lambda_bml", cfg.training.lambda_bml, float)
    cfg.training.warm_up_step = safe_input("Enter warmup steps", cfg.training.warm_up_step, int)
    cfg.training.update_step = safe_input("Enter update step interval", cfg.training.update_step, int)
    cfg.training.num_workers = safe_input("Enter number of workers", cfg.training.num_workers, int)

    resume = input(f"Resume from checkpoint? (yes/no, default {'yes' if cfg.training.resume_checkpoint else 'no'}): ").strip().lower()
    if resume:
        cfg.training.resume_checkpoint = True if resume == "yes" else False
    else:
        loss = None    

    # === Assertions for required paths ===
    #required_paths = ["csv_path", "json_path", "output_base"]
    #for path_key in required_paths:
     #   if not cfg.paths.get(path_key):
      #      raise ValueError(f"Missing required path: `{path_key}`. Please set it in your `local.yaml`.")

    #cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = build_run_name(cfg)

    # === Setup output directories ===
    os.makedirs(cfg.paths.model_save_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    
  
    cfg.paths.save_model_dir = os.path.join(cfg.paths.model_save_dir, run_name)
    cfg.paths.best_model_dir = os.path.join(cfg.paths.checkpoint_dir, run_name)
    cfg.paths.loss_curve_dir = os.path.join(cfg.paths.output_base, "training_loss", run_name)
    cfg.paths.tensorboard_dir = os.path.join(cfg.paths.output_base, "tensorboard", run_name)
    #cfg.paths.umap_dir = os.path.join(cfg.paths.output_base, "figures", "queue_umaps_training", run_name)
    cfg.paths.plot_dir = os.path.join(cfg.paths.output_base, "figures", "training_loss_curve", run_name)
    cfg.paths.live_loss_plot_dir = os.path.join(cfg.paths.output_base, "figures", "live_loss_plot", run_name)
    cfg.paths.epoch_loss_plot_dir = os.path.join(cfg.paths.output_base, "figures", "epoch_loss_plot", run_name)
    cfg.paths.csv_report_dir = os.path.join(cfg.paths.output_base, "training_csv_reports", run_name)
    cfg.paths.lr_plot = os.path.join(cfg.paths.output_base, "figures", "lr_plots", run_name)

    for p in [
        cfg.paths.plot_dir, cfg.paths.save_model_dir, cfg.paths.best_model_dir,
        cfg.paths.tensorboard_dir, cfg.paths.live_loss_plot_dir,
        cfg.paths.loss_curve_dir, cfg.paths.csv_report_dir, cfg.paths.lr_plot
    ]:
        os.makedirs(p, exist_ok=True)

    # === Logger setup ===
    #logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {cfg.training.device}")
    logger.info(f"Run name: {run_name}")

    # === Prepare model, dataloader, loss ===
    device = torch.device(cfg.training.device)
    dataloader = get_dataloader(cfg, use_neighbors=cfg.training.use_neighbors)
    model = get_model(cfg, dataloader).to(device)
    criterion = get_loss_fn(cfg)
    writer = SummaryWriter(log_dir=cfg.paths.tensorboard_dir)
    # === Start training ===
    train_moco(cfg=cfg, model=model, dataloader=dataloader, criterion=criterion, loss=loss, writer=writer, run_name=run_name)