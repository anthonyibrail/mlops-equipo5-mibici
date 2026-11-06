from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml

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

# Configuracion del proyecto (problema, validacion, modelos): ver params.yaml
PARAMS_PATH = PROJ_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Lee params.yaml. Las rutas de la seccion 'datos' se devuelven absolutas."""
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["datos"] = {k: PROJ_ROOT / v for k, v in params.get("datos", {}).items()}
    return params


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
