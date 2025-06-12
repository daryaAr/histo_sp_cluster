from histo_sp_cluster_SSL.moco import build_resnet_with_projection
from histo_sp_cluster_SSL.utils.logger import logger
import torch
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

def initialize_queue_random_from_dataset(cfg, dataloader, samples_per_batch=15):
    """
    Initializes the MoCo memory queue using random samples from the dataset and passes them through a base encoder.

    Args:
        cfg: Configuration object.
        dataloader: PyTorch DataLoader to sample from.
        samples_per_batch: Number of random samples to select from each batch.

    Returns:
        For moco_superpixel_cluster_bioptimus:
            - queue: Tensor [queue_size, D]
            - cluster_ids: Tensor [queue_size, 3]
        For other types:
            - queue: Tensor [queue_size, D]
    """
    
    logger.info("Initializing memory queue from dataset...")

    base_encoder_name = cfg.model.base_encoder
    output_dim = cfg.model.output_dim
    queue_size = cfg.model.queue_size
    device = cfg.training.device
    moco_type = cfg.model.moco_type

    # Build encoder
    encoder = build_resnet_with_projection(base_encoder_name, output_dim, pretrained=True).to(device)
    encoder.eval()

    
    cluster_ids = []
    total_collected = 0
    batch_queue = []
    batch_cluster_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling for initial queue"):
           
            if moco_type == "moco_superpixel_cluster_bioptimus":
                imgs, _, _, anchor_clusters, _ = batch
            elif moco_type in ["moco_superpixel", "moco_superpixel_cluster"]:
                imgs, _, _ = batch
            else:  # moco_v2
                imgs, _ = batch

            batch_size = imgs.size(0)
            selected_indices = random.sample(range(batch_size), min(samples_per_batch, batch_size))

            # Get selected images
            batch_images = imgs[selected_indices].to(device)
            batch_features = F.normalize(encoder(batch_images), dim=1)
            batch_queue.append(batch_features)


            if moco_type == "moco_superpixel_cluster_bioptimus":
                batch_clusters = anchor_clusters[selected_indices].to(device)
                batch_cluster_ids.append(batch_clusters)

            total_collected += batch_features.size(0)
            if total_collected >= queue_size:
                break

      # Concatenate and trim to exact queue size
    queue = torch.cat(batch_queue, dim=0)[:queue_size]

    if moco_type == "moco_superpixel_cluster_bioptimus":
        cluster_ids_tensor = torch.cat(batch_cluster_ids, dim=0)[:queue_size]
        logger.info(f"Queue shape: {queue.shape}, Cluster shape: {cluster_ids_tensor.shape}")
        return queue, cluster_ids_tensor
    else:
        logger.info(f"Queue shape: {queue.shape}")
        return queue