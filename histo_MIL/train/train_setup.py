import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from histo_MIL.utils.logger import logger
from histo_MIL.utils.training_utils import (
    save_checkpoint,
    save_mil_losses,
    save_confusion_matrix,
    log_mil_training_run,
    plot_train_val_loss
)
from histo_MIL.utils.loss import (
        cross_entropy_loss_fn,
        get_accuracy_metric,
        get_f1_metric,
        get_confusion_matrix_metric
    )

def train_mil(cfg, model, train_loader, val_loader, writer, run_name):
   

    
    logger.info("Starting MIL training...")
    device = torch.device(cfg.training.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    acc_metric = get_accuracy_metric(num_classes=cfg.model.num_classes)
    f1_metric = get_f1_metric(num_classes=cfg.model.num_classes)
    conf_matrix_metric = get_confusion_matrix_metric(num_classes=cfg.model.num_classes)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.training.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs} started.")
        model.train()

        train_loss_total = 0.0
        all_train_preds, all_train_labels = [], []

        for batch_idx, (bags, lengths, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_preds, batch_labels = [], []

            for i in range(len(bags)):  # Loop over bags in the batch
                bag = bags[i, :lengths[i]].to(device)  # shape: [num_tiles, input_dim]
                label = labels[i].unsqueeze(0).to(device)

                with torch.amp.autocast(device_type="cuda"):
                    output, _ = model(bag)  # output shape: [num_classes]
                    loss = cross_entropy_loss_fn(output.unsqueeze(0), label)

                scaler.scale(loss).backward()
                batch_loss += loss.item()
                batch_preds.append(output.argmax().unsqueeze(0).detach().cpu())
                batch_labels.append(label.detach().cpu())

            scaler.step(optimizer)
            scaler.update()

            train_loss_total += batch_loss
            all_train_preds.extend(batch_preds)
            all_train_labels.extend(batch_labels)

            if batch_idx+1 == 20:
                break
        

        avg_train_loss = train_loss_total / len(train_loader)
        train_losses.append(avg_train_loss)
        train_preds = torch.cat(all_train_preds)
        train_labels = torch.cat(all_train_labels)
        train_acc = acc_metric(train_preds, train_labels).item()
        train_f1 = f1_metric(train_preds, train_labels).item()
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        logger.info(f"[Train] Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")

        # -------- Validation --------
        model.eval()
        val_loss_total = 0.0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for batch_idx, (bags, lengths, labels) in enumerate(tqdm(val_loader, desc="Validation")):
                batch_loss = 0.0
                batch_preds, batch_labels = [], []

                for i in range(len(bags)):
                    bag = bags[i, :lengths[i]].to(device)
                    label = labels[i].unsqueeze(0).to(device)

                    with torch.amp.autocast(device_type="cuda"):
                        output, _ = model(bag)
                        loss = cross_entropy_loss_fn(output.unsqueeze(0), label)

                    batch_loss += loss.item()
                    batch_preds.append(output.argmax().unsqueeze(0).cpu())
                    batch_labels.append(label.cpu())

                val_loss_total += batch_loss
                all_val_preds.extend(batch_preds)
                all_val_labels.extend(batch_labels)

        avg_val_loss = val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)
        val_preds = torch.cat(all_val_preds)
        val_labels = torch.cat(all_val_labels)
        val_acc = acc_metric(val_preds, val_labels).item()
        val_f1 = f1_metric(val_preds, val_labels).item()
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)


        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        logger.info(f"[Val] Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        # Save confusion matrix
        cm_tensor = conf_matrix_metric(val_preds, val_labels)
        cm_path = Path(cfg.mil_paths.confusion_dir) / f"confusion_matrix_val_epoch_{epoch+1}.png"
        save_confusion_matrix(cm_tensor, class_names=list(cfg.mil.label_mapping.keys()), save_path=cm_path, epoch=epoch+1)

         
        save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                base_lr=cfg.training.learning_rate,
                checkpoint_path=Path(cfg.mil_paths.model_save_dir) / f"epoch_{epoch}.pth",
                best=False
            )
            

        # Best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                base_lr=cfg.training.learning_rate,
                checkpoint_path=Path(cfg.mil_paths.checkpoint_dir) / f"best_model_{run_name}.pth",
                best=True
            )
            logger.info("Saved best model.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation loss. Patience: {patience_counter}/{cfg.training.early_stopping_patience}")

        if cfg.training.early_stopping and patience_counter >= cfg.training.early_stopping_patience:
            logger.info("Early stopping triggered.")
            break

    writer.close()

    save_mil_losses(train_losses, train_accuracies, train_f1s, Path(cfg.mil_paths.loss_log_dir) / "train")
    save_mil_losses(val_losses, val_accuracies, val_f1s, Path(cfg.mil_paths.loss_log_dir) / "val")


    log_mil_training_run(
        cfg=cfg,
        run_name=run_name,
        train_loss=train_losses[-1],
        train_acc=train_acc,
        train_f1=train_f1,
        val_loss=val_losses[-1],
        val_acc=val_acc,
        val_f1=val_f1
    )
    plot_train_val_loss(
    train_losses=train_losses,
    val_losses=val_losses,
    ylabel="Loss",
    save_path=Path(cfg.mil_paths.loss_plot_dir) / "mil_loss_curve.png"
    )
    plot_train_val_loss(
    train_losses=train_accuracies,
    val_losses=val_accuracies,
    ylabel="Accuracy",
    save_path=Path(cfg.mil_paths.metric_plot_dir) / "mil_accuracy_curve.png"
    )
    plot_train_val_loss(
    train_losses=train_f1s,
    val_losses=val_f1s,
    ylabel="F1 score",
    save_path=Path(cfg.mil_paths.metric_plot_dir) / "mil_f1_curve.png"
    )

    logger.info("Training completed.")