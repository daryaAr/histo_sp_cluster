from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
#from loguru import logger
from histo_MIL.utils.logger import logger 

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"




def load_yaml_config(
    default_path: str = "configs/mil_default.yaml",
    local_path: str = "configs/local.yaml"
):
    """
    Load configuration from default and local YAML files.

    Args:
        default_path (str): Path to the default config.
        local_path (str): Path to the local override config.

    Returns:
        OmegaConf.DictConfig: Merged configuration object.
    """
    proj_root = Path(__file__).resolve().parents[1]
    default_path = proj_root / default_path
    local_path = proj_root / local_path

    default_cfg = OmegaConf.load(default_path)
    try:
        local_cfg = OmegaConf.load(local_path)
        cfg = OmegaConf.merge(default_cfg, local_cfg)
    except FileNotFoundError:
        logger.warning(f"No local config found at {local_path}. Using default only.")
        cfg = default_cfg

    return cfg