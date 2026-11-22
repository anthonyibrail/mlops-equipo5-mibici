"""Carga de modelos para el servicio MiBici.

Carga los pipelines pkl desde models/ al arrancar el servicio.
Implementa un singleton para no recargar en cada request.

bool_to_int se importa desde mibici.modeling (no desde __main__)
para que pickle.load funcione correctamente desde este modulo.
El path estable es mibici.modeling.bool_to_int.
"""

from __future__ import annotations

import pickle

from loguru import logger
from sklearn.pipeline import Pipeline

from mibici.config import MODELS_DIR
from mibici.modeling import bool_to_int  # noqa: F401 -- necesario para pickle

_pipelines: dict[str, Pipeline] | None = None


def load_pipelines() -> dict[str, Pipeline]:
    """Carga y retorna los pipelines pkl desde models/.

    Returns:
        dict con claves "salidas" y "llegadas", cada una con su Pipeline.

    Raises:
        FileNotFoundError: si algun pkl no existe en models/.
    """
    pipelines: dict[str, Pipeline] = {}
    for target in ("salidas", "llegadas"):
        path = MODELS_DIR / f"pipeline_{target}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {path}. "
                "Ejecuta 'dvc repro train' o 'make evaluate register promote' primero."
            )
        with open(path, "rb") as fh:
            pipelines[target] = pickle.load(fh)
        logger.info(f"Pipeline '{target}' cargado desde {path}")
    return pipelines


def get_pipelines() -> dict[str, Pipeline]:
    """Retorna el singleton de pipelines, cargandolo si no existe.

    Returns:
        dict con claves "salidas" y "llegadas".
    """
    global _pipelines
    if _pipelines is None:
        _pipelines = load_pipelines()
    return _pipelines
