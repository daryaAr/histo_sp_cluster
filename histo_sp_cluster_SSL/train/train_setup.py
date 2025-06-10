import os
import time
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from histo_sp_cluster_SSL.config import load_yaml_config
from histo_sp_cluster_SSL.dataloader import get_dataloader
from histo_sp_cluster_SSL.Loss import ClusterLoss, ContrastiveLoss

from histo_sp_cluster_SSL.utils.training_utils import (
    adjust_learning_rate, update_template_csv, save_checkpoint,
    get_latest_checkpoint, load_existing_losses, save_losses, log_training_run, save_losses_cluster
)
from histo_sp_cluster_SSL.utils.visualization import plot_loss_curve, plot_epoch_losses, plot_lr_schedule, plot_cluster_losses
from histo_sp_cluster_SSL.utils.logger import logger


def train_moco(cfg, model, dataloader, criterion, loss, writer, run_name):
    #logger = cfg.logger
    logger.info("Starting training...")
    start_time = datetime.now()
    device = torch.device(cfg.training.device)


    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    step = 0
    start_epoch = 0
    best_loss = float("inf")
    warm_up_step = 100
    loss_log = load_existing_losses(cfg.paths.loss_curve_dir)
    contrastive = ContrastiveLoss(temperature=cfg.model.temperature)

    losses_all = []
    contrastive_losses = []
    fn_losses = []
    neighbor_losses = []
    lr_history = []

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
        lr_history.append(lr)
        logger.info(f"Learning rate adjusted to: {lr:.6f}")
        losses_epoch_all = []
        contrastive_epoch_losses = []
        fn_epoch_losses = []
        neighbor_epoch_losses = []

        #data_start_time = time.perf_counter()
        #epoch_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(dataloader):
            #batch_time_start = time.perf_counter()
            logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}, Batch {batch_idx + 1}/{len(dataloader)}")

            if cfg.model.moco_type == "moco_v2":
                images_q, images_k = batch
                images_q = images_q.to(device, non_blocking=True)
                images_k = images_k.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    q, k = model(images_q, images_k)
                    loss = criterion(q, k, model.queue)
                    logger.info(f"Loss: {loss.item():.4f}")
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
                    logger.info(f"Loss: {loss.item():.4f}, Tile Loss: {loss_tile.item():.4f}, Neighbor Loss: {loss_neighbor.item():.4f}")
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
                      
            elif cfg.model.moco_type == "moco_superpixel_cluster_bioptimus":
                images_q, images_k1, images_k2, images_q_cluster, images_k2_cluster = batch
                images_q = images_q.to(device, non_blocking=True)
                images_k1 = images_k1.to(device, non_blocking=True)
                images_k2 = images_k2.to(device, non_blocking=True)
                images_q_cluster = images_q_cluster.to(device, non_blocking=True)
                images_k2_cluster = images_k2_cluster.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    q, k1, k2, fn_em, hn_em, _ = model(
                        images_q, images_k1, images_k2, images_q_cluster
                    )
                    loss, contrastive_loss, fn_loss, neighbor_loss = criterion(q, k1, k2, hn_em, fn_em)
                    logger.info(f"Loss: {loss.item():.4f}, Contrastive Loss: {contrastive_loss.item():.4f}, Neighbor Loss: {neighbor_loss.item():.4f}, False-Negative Loss: {fn_loss.item():.4f}")

            else:
                raise ValueError(f"Unsupported MoCo type: {cfg.model.moco_type}")

            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

                

            if (cfg.model.moco_type == "moco_superpixel_cluster") and (step+1 > warm_up_step):
                    
                losses_epoch_all.append(loss.item())
                contrastive_epoch_losses.append(contrastive_loss.item())
                fn_epoch_losses.append(fn_loss.item())
                neighbor_epoch_losses.append(neighbor_loss.item())
                

            elif (cfg.model.moco_type == "moco_superpixel_cluster_bioptimus"):
                    
                losses_epoch_all.append(loss.item())
                contrastive_epoch_losses.append(contrastive_loss.item())
                fn_epoch_losses.append(fn_loss.item())
                neighbor_epoch_losses.append(neighbor_loss.item())
                      


            if hasattr(model, "update_queue"):
                if cfg.model.moco_type == "moco_superpixel_cluster":
                    model.update_queue(k1, k2, k1_ids, k2_ids)
                elif cfg.model.moco_type == "moco_superpixel":
                    model.update_queue(k1, k2)
                elif cfg.model.moco_type == "moco_superpixel_cluster_bioptimus": 
                    model.update_queue(k1, k2, images_q_cluster, images_k2_cluster)
                else:
                    model.update_queue(k)
            step += 1 
            #data_start_time = time.perf_counter()
            #if batch_idx+1 == 20:
              #  break   

            torch.cuda.empty_cache()    

            
            
       
        if (cfg.model.moco_type == "moco_superpixel_cluster_bioptimus"):    
            plot_epoch_losses(
                losses_all=losses_epoch_all,
                contrastive_losses=contrastive_epoch_losses,
                fn_losses=fn_epoch_losses,
                neighbor_losses=neighbor_epoch_losses,
                epoch=epoch+1,
                save_dir=cfg.paths.live_loss_plot_dir
            )
                    
             

        if (cfg.model.moco_type == "moco_superpixel_cluster_bioptimus") or (cfg.model.moco_type == "moco_superpixel_cluster"):
            avg_epoch_loss = sum(losses_epoch_all) / len(losses_epoch_all)
            avg_epoch_contrastive_loss = sum(contrastive_epoch_losses) / len(contrastive_epoch_losses)
            avg_epoch_fn_losses = sum(fn_epoch_losses) / len(fn_epoch_losses)    
            avg_epoch_neighbor_losses = sum(neighbor_epoch_losses) / len(neighbor_epoch_losses)
            losses_all.append(avg_epoch_loss)
            contrastive_losses.append(avg_epoch_contrastive_loss)
            fn_losses.append(avg_epoch_fn_losses)
            neighbor_losses.append(avg_epoch_neighbor_losses)
            writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
            writer.add_scalar("Contrastive_Loss/train", avg_epoch_contrastive_loss, epoch)
            writer.add_scalar("fn_Loss/train", avg_epoch_fn_losses, epoch) 
            writer.add_scalar("Neighbor_Loss/train", avg_epoch_neighbor_losses, epoch)
            save_losses_cluster(losses_all, contrastive_losses, fn_losses, neighbor_losses, cfg.paths.loss_curve_dir)
        else: 
            avg_epoch_loss = running_loss / num_batches if num_batches > 0 else float("inf")
            loss_log.append(avg_epoch_loss)
            writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
            save_losses(loss_log, cfg.paths.loss_curve_dir)       
            

        save_checkpoint(epoch, model, optimizer, scaler, cfg.training.learning_rate, step, os.path.join(cfg.paths.save_model_dir, f"epoch_{epoch}.pth"))
        
       
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(epoch, model, optimizer, scaler, cfg.training.learning_rate, step, os.path.join(cfg.paths.best_model_dir, f"best_model.pth"), best=True)

    writer.close()
    elapsed = datetime.now() - start_time

    if (cfg.model.moco_type == "moco_superpixel_cluster_bioptimus") or (cfg.model.moco_type == "moco_superpixel_cluster"):
        avg_loss = sum(losses_all)/len(losses_all)
        #avg_contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
        #avg_fn_loss = sum(fn_losses)/len(fn_losses)
        #avg_neighbor_loss = sum(neighbor_losses)/len(neighbor_losses)
        log_training_run(cfg, training_loss=avg_loss, run_name=run_name)
        plot_cluster_losses(losses_all, contrastive_losses, fn_losses, neighbor_losses, save_dir=cfg.paths.plot_dir)
    else:
        avg_loss=sum(loss_log)/len(loss_log)
        log_training_run(cfg, training_loss=avg_loss, run_name=run_name)
        plot_loss_curve(loss_log, save_path=os.path.join(cfg.paths.plot_dir, "training_loss_curve.png"))

    # Plot final loss curve
    
    plot_lr_schedule(lr_history=lr_history,save_dir=cfg.paths.lr_plot)
   

    logger.info(f"Training finished. Total time: {elapsed}")
