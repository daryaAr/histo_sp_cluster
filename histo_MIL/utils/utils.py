def build_run_name(cfg):
    return f"MIL_{cfg.embeddings.type}_{cfg.mil.mode}_bs{cfg.training.batch_size}_lr{cfg.training.learning_rate}_epochs{cfg.training.epochs}"

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
        "moco": 128,
        "superpixel+cluster": 128,
        "superpixel": 128,
    }

    embedding_type = cfg.embeddings.type.lower()
    if embedding_type not in dim_map:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    cfg.embeddings.dim = dim_map[embedding_type]