"""Inferencia de demanda horaria con los pipelines entrenados (Fase 2).

Carga los pipelines guardados en models/ por train.py y genera predicciones
de salidas y llegadas para el periodo de prueba (o cualquier panel v3).
El resultado incluye los valores reales y los pronosticados para facilitar
la evaluacion y comparacion en MLflow.

Uso:
    python -m mibici.modeling.predict
    python -m mibici.modeling.predict --cutoff 2026-04
    python -m mibici.modeling.predict --output-path data/processed/predicciones.parquet
"""

from __future__ import annotations

import pickle
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import MODELS_DIR, PROCESSED_DATA_DIR, load_params
from mibici.modeling.train import ALL_FEATURES

app = typer.Typer()


def load_pipeline(target: str, models_dir: Path = MODELS_DIR):
    """Carga el pipeline serializado para el objetivo indicado.

    Args:
        target:     Nombre del objetivo ('salidas' o 'llegadas').
        models_dir: Directorio donde train.py guarda los pipelines.

    Returns:
        Pipeline de sklearn cargado desde disco.

    Raises:
        FileNotFoundError: si el pipeline no existe (train.py aun no se ejecuto).
    """
    model_path = models_dir / f"pipeline_{target}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el pipeline en {model_path}. "
            "Ejecuta primero: python -m mibici.modeling.train"
        )
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "predicciones.parquet",
    cutoff: str = typer.Option(
        None,
        help="YYYY-MM: inicio del periodo de prueba. Default: params.yaml",
    ),
) -> None:
    """Genera predicciones de salidas y llegadas y las guarda en un parquet."""
    params = load_params()
    if cutoff is None:
        cutoff = params["validacion"]["primer_mes_prueba"]
    targets = params["problema"]["objetivos"]

    logger.info(f"Leyendo panel v3 desde {params['datos']['features']}...")
    df = pd.read_parquet(params["datos"]["features"])

    cutoff_ts = pd.Timestamp(cutoff + "-01")
    test = df[df["hora_inicio"] >= cutoff_ts].copy()
    logger.info(f"Periodo de prueba (desde {cutoff}): {len(test):,} filas")

    for target in targets:
        logger.info(f"Cargando pipeline y generando predicciones para '{target}'...")
        pipeline = load_pipeline(target)
        pred = np.clip(pipeline.predict(test[ALL_FEATURES]), 0.0, None)
        test[f"{target}_pred"] = pred

    # Columnas de salida: identificadores + reales + predichos
    id_cols = ["hora_inicio", "estacion_id"]
    value_cols = []
    for t in targets:
        value_cols += [t, f"{t}_pred"]
    result = test[id_cols + value_cols].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.success(f"Predicciones guardadas en {output_path} ({len(result):,} filas)")


if __name__ == "__main__":
    app()
