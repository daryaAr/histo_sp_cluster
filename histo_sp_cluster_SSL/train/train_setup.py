import os
import time
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from histo_sp_cluster_SSL.config import load_yaml_config
from histo_sp_cluster_SSL.dataloader import get_dataloader
from histo_sp_cluster_SSL.Loss import ClusterLoss, ContrastiveLoss
from histo_sp_cluster_SSL.moco import (
    MoCoV2Encoder,
    MoCoSuperpixel,
    MoCoSuperpixelCluster
)
from histo_sp_cluster_SSL.utils.training_utils import (
    adjust_learning_rate, update_template_csv, save_checkpoint,
    get_latest_checkpoint, load_existing_losses, save_losses
)
from histo_sp_cluster_SSL.utils.visualization import plot_loss_curve, plot_live_losses
from histo_sp_cluster_SSL.utils.logger import logger


def train_moco(cfg, model, dataloader, criterion, loss, writer, run_name):
    #logger = cfg.logger
    logger.info("Starting training...")
    start_time = datetime.now()
    device = torch.device(cfg.training.device)

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_save_dir, exist_ok=True)
    os.makedirs(cfg.paths.plot_dir, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    step = 0
    start_epoch = 0
    best_loss = float("inf")
    warm_up_step = 100
    loss_log = load_existing_losses(cfg.paths.plot_dir)
    contrastive = ContrastiveLoss(temperature=cfg.model.temperature)

    x_labels = []
    losses_all = []
    contrastive_losses = []
    fn_losses = []
    neighbor_losses = []

    if cfg.training.resume_checkpoint:
        ckpt_path = get_latest_checkpoint(cfg.paths.model_save_dir)
        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            step = checkpoint['current_step']
            logger.info(f"Resumed from epoch {start_epoch}, step {step}")

    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs} started.")
        model.train()
        running_loss = 0.0
        num_batches = 0

        lr = adjust_learning_rate(optimizer, epoch, cfg.training.learning_rate, cfg.training.epochs)
        logger.info(f"Learning rate adjusted to: {lr:.6f}")

        data_start_time = time.perf_counter()
        epoch_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(dataloader):
            batch_time_start = time.perf_counter()

            if cfg.model.moco_type == "moco_v2":
                images_q, images_k = batch
                images_q = images_q.to(device, non_blocking=True)
                images_k = images_k.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    q, k = model(images_q, images_k)
                    loss = criterion(q, k, model.queue)
            elif cfg.model.moco_type == "moco_superpixel":
                images_q, images_k1, images_k2 = batch
                images_q = images_q.to(device, non_blocking=True)
                images_k1 = images_k1.to(device, non_blocking=True)
                images_k2 = images_k2.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    q, k1, k2 = model(images_q, images_k1, images_k2)
                    loss_tile = criterion(q, k1, model.queue)
                    loss_neighbor = criterion(q, k2, model.queue)
                    loss = cfg.loss.alpha * loss_tile + (1 - cfg.loss.alpha) * loss_neighbor
            elif cfg.model.moco_type == "moco_superpixel_cluster":
                images_q, images_k1, images_k2 = batch
                images_q = images_q.to(device, non_blocking=True)
                images_k1 = images_k1.to(device, non_blocking=True)
                images_k2 = images_k2.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    q, k1, k2, fn_em, hn_em, fn_in, hn_in, q_ids, k1_ids, k2_ids, queue = model(
                        images_q, images_k1, images_k2, step=step + 1, update_step=cfg.training.update_step
                    )
                    if step+1 <= warm_up_step:
                        loss_tile = contrastive(q, k1, model.queue)
                        loss_neighbor = contrastive(q, k2, model.queue)
                        loss = 0.5 * loss_tile + (1 - 0.5) * loss_neighbor
                    else:
                        model.cluster_helper.update_centroids(
                        memory_bank=model.queue, step=step+1, update_step=cfg.training.update_step
                        )
                        loss, contrastive_loss, fn_loss, neighbor_loss = criterion(q, k1, k2, hn_em, fn_em)

                        if (step % cfg.training.update_step == 0) and model.cluster_helper.initialized:
                            memory_emb, memory_ids, centroids = model.cluster_helper.get_memory_bank_clusters()
                    
                        logger.info(f"Loss: {loss.item():.4f}, Contrastive Loss: {contrastive_loss.item():.4f}, Neighbor Loss: {neighbor_loss.item():.4f}, False-Negative Loss: {fn_loss.item():.4f}")
                        """
                        if memory_emb is not None and memory_ids is not None:
                            plot_umap_with_queries(
                                memory_bank_embeddings=memory_emb.cpu().numpy(),
                                memory_bank_cluster_ids=memory_ids.cpu().numpy(),
                                centroids=centroids.cpu().numpy(),
                                queries=q.cpu().numpy(),
                                query_cluster_ids=q_ids.cpu().numpy(),
                                save_path=os.path.join(cfg.paths.plot_dir, f"umap_{epoch+1}_step{step+1}.png"),
                                annotate_query_ids=True
                            )
                        """
            else:
                raise ValueError(f"Unsupported MoCo type: {cfg.model.moco_type}")

            # Backward + optimize
            if loss != None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1

                torch.cuda.empty_cache()
                

                if (cfg.model.moco_type == "moco_superpixel_cluster") and (step+1 > warm_up_step):
                    
                    
                    losses_all.append(loss.item())
                    contrastive_losses.append(contrastive_loss.item())
                    fn_losses.append(fn_loss.item())
                    neighbor_losses.append(neighbor_loss.item())

                    # Plot live loss every 200 batches
                    if (step + 1) % 400 == 0:
                        plot_live_losses(
                            losses_all=losses_all,
                            contrastive_losses=contrastive_losses,
                            fn_losses=fn_losses,
                            neighbor_losses=neighbor_losses,
                            save_dir=cfg.paths.live_loss_plot_dir,
                            epoch=epoch+1,
                            batch_idx=batch_idx+1
                        )

        


            if hasattr(model, "update_queue"):
                if cfg.model.moco_type == "moco_superpixel_cluster":
                    model.update_queue(k1, k2, k1_ids, k2_ids)
                elif cfg.model.moco_type == "moco_superpixel":
                    model.update_queue(k1, k2)
                else:
                    model.update_queue(k)

            
            step += 1

            logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}, Batch {batch_idx + 1}/{len(dataloader)}")

            data_start_time = time.perf_counter()
            #if batch_idx+1 == 20:
              #  break

        avg_epoch_loss = running_loss / num_batches if num_batches > 0 else float("inf")
        loss_log.append(avg_epoch_loss)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)

        save_checkpoint(epoch, model, optimizer, scaler, cfg.training.learning_rate, step, os.path.join(cfg.paths.save_model_dir, f"epoch_{epoch}.pth"))
        save_losses(loss_log, cfg.paths.loss_curve_dir)

       
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(epoch, model, optimizer, scaler, cfg.training.learning_rate, step, os.path.join(cfg.paths.best_model_dir, f"best_model.pth"), best=True)

    writer.close()

    elapsed = datetime.now() - start_time
    update_template_csv(
        csv_path=os.path.join(cfg.paths.training_csv_dir, f"{run_name}.csv"),
        batch_size=cfg.training.batch_size,
        temperature=cfg.training.temperature,
        avg_loss=sum(loss_log)/len(loss_log),
        train_time=str(elapsed),
        num_epochs=cfg.training.epochs
    )

    # Plot final loss curve
    plot_loss_curve(loss_log, save_path=os.path.join(cfg.paths.loss_curve_dir, "training_loss_curve.png"))

    logger.info(f"Training finished. Total time: {elapsed}")
