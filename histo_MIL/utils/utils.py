def build_run_name(cfg):
    if cfg.training.bag is not None:
        return f"MIL_{cfg.embeddings.type}_{cfg.mil.mode}_{cfg.embeddings.dim}_lr{cfg.training.learning_rate}_bs{cfg.training.batch_size}_bag{cfg.training.bag}"
    else:
        return f"MIL_{cfg.embeddings.type}_{cfg.mil.mode}_{cfg.embeddings.dim}_lr{cfg.training.learning_rate}_bs{cfg.training.batch_size}_no_limit"

def set_embedding_dim(cfg):
    """
    Automatically sets cfg.embeddings.dim based on cfg.embeddings.type.

    Supported types and their dimensions:
        - resnet50: 2048
        - bioptimus: 1536
        - moco: 128
        - superpixel+cluster: 128
        - superpixel: 128
    """
    dim_map = {
        "resnet50": 2048,
        "bioptimus": 1536,
        "mocov2": 2048,
        "superpixel_cluster": 2048,
        "superpixel": 2048,
    }

    embedding_type = cfg.embeddings.type.lower()
    if embedding_type not in dim_map:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    cfg.embeddings.dim = dim_map[embedding_type]