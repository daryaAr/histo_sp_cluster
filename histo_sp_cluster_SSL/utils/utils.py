from histo_sp_cluster_SSL.moco import (
    MoCoV2Encoder,
    MoCoSuperpixel,
    MoCoSuperpixelCluster,
    MoCoSuperpixelClusterBioptimus
)
from histo_sp_cluster_SSL.queue_init import initialize_queue_random_from_dataset
from histo_sp_cluster_SSL.Loss import ClusterLoss, ContrastiveLoss


def get_model(cfg, dataloader):
    model_cls_map = {
        "moco_v2": MoCoV2Encoder,
        "moco_superpixel": MoCoSuperpixel,
        "moco_superpixel_cluster": MoCoSuperpixelCluster,
        "moco_superpixel_cluster_bioptimus": MoCoSuperpixelClusterBioptimus
    }

    init_queue, init_cluster_ids = None, None
    if cfg.model.init_queue_type == "awared_random":
      result = initialize_queue_random_from_dataset(cfg, dataloader, samples_per_batch=15)
      if isinstance(result, tuple):
        init_queue, init_cluster_ids = result
      else:
        init_queue = result

    model_type = cfg.model.moco_type.lower()
    model_cls = model_cls_map[model_type]

    common_kwargs = dict(
        base_encoder=cfg.model.base_encoder,
        output_dim=cfg.model.output_dim,
        queue_size=cfg.model.queue_size,
        momentum=cfg.model.momentum,
        temperature=cfg.model.temperature,
        init_queue=init_queue
    )

    if model_type == "moco_superpixel_cluster":
        return model_cls(
            **common_kwargs,
            num_clusters=cfg.model.num_clusters,
            device=cfg.training.device
        )
    elif model_type == "moco_superpixel_cluster_bioptimus":
        return model_cls(
            **common_kwargs,
            init_cluster_ids=init_cluster_ids
        )
    else:
        return model_cls(**common_kwargs)

def get_loss_fn(cfg):
    if cfg.loss.type == "contrastive":
        return ContrastiveLoss(temperature=cfg.model.temperature)
    elif cfg.loss.type == "cluster":
        return ClusterLoss(
            temperature=cfg.model.temperature,
            alpha=cfg.training.alpha,
            beta=cfg.training.beta,
            lambda_bml=cfg.training.lambda_bml
        )
    else:
        raise ValueError(f"Unsupported loss type: {cfg.loss.type}")

def build_run_name(cfg):
    return f"{cfg.cluster.cluster_type}_queue_wo_fn_queue{cfg.model.queue_size}_alpha{cfg.training.alpha}_beta{cfg.training.beta}_epochs{cfg.training.epochs}_{cfg.model.moco_type}"        