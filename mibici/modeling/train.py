"""Entrenamiento de modelos de pronostico de demanda (Fase 2).

Para cada objetivo (salidas, llegadas) construye y entrena un Pipeline de
scikit-learn que encapsula el preprocesamiento con ColumnTransformer y el
modelo de estimacion. El pipeline entrenado se guarda en models/ y se
registra en MLflow.

Estructura del Pipeline:
  - ColumnTransformer:
      * "num": passthrough para features numericos
               (HistGBR maneja NaN de lags del inicio de la serie nativamente).
      * "bool": cast a int para flags booleanos.
  - HistGradientBoostingRegressor con loss=poisson, adecuado para conteos >= 0.

Uso:
    python -m mibici.modeling.train
    python -m mibici.modeling.train --cutoff 2026-04
"""

from __future__ import annotations

import pickle

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import typer

from mibici.config import MODELS_DIR, load_params
from mibici.modeling import bool_to_int
from mibici.modeling.validation import regression_metrics
from mibici.versioning import TRACKING_URI

app = typer.Typer()

# --- Columnas que entran al modelo (panel v3) ---
# Numericas: HistGBR las acepta con NaN nativo (lags al inicio de la serie).
NUMERIC_FEATURES = [
    "hora",
    "dia_semana",
    "mes",
    "salidas_lag_24h",
    "salidas_lag_48h",
    "salidas_lag_168h",
    "llegadas_lag_24h",
    "llegadas_lag_168h",
    "media_7d",
    "salidas_sistema_lag_24h",
    "capacidad",
    "lat",
    "lon",
]
# Booleanas: se convierten a int (0/1) antes de pasar al estimador.
BOOL_FEATURES = [
    "es_fin_de_semana",
    "es_hora_pico_manana",
    "es_hora_pico_tarde",
]
ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES


def build_pipeline(model_cfg: dict) -> Pipeline:
    """Construye el Pipeline de sklearn: ColumnTransformer + HistGBR.

    El preprocesador asigna transformaciones distintas segun el tipo de
    columna; el estimador recibe todos los features como un array numerico.

    Args:
        model_cfg: Seccion 'fijos' del modelo en params.yaml
                   (p. ej. {'loss': 'poisson', 'max_iter': 300}).

    Returns:
        Pipeline sin entrenar listo para llamar a fit().
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("bool", FunctionTransformer(bool_to_int), BOOL_FEATURES),
        ],
        remainder="drop",
    )
    estimator = HistGradientBoostingRegressor(**model_cfg)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def temporal_split(
    df: pd.DataFrame,
    cutoff: str,
    time_col: str = "hora_inicio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el panel en entrenamiento y prueba por fecha.

    Los datos anteriores al cutoff van a entrenamiento; el resto a prueba.
    No hay mezcla de periodos: garantia de no-filtracion temporal.

    Args:
        df:       Panel de features v3.
        cutoff:   Fecha de corte en formato 'YYYY-MM'. El primer dia de ese
                  mes es el inicio del conjunto de prueba.
        time_col: Columna datetime del panel.

    Returns:
        (train, test): tupla de DataFrames.
    """
    cutoff_ts = pd.Timestamp(cutoff + "-01")
    train = df[df[time_col] < cutoff_ts]
    test = df[df[time_col] >= cutoff_ts]
    logger.info(f"Corte {cutoff}: entrenamiento {len(train):,} filas | prueba {len(test):,} filas")
    return train, test


@app.command()
def main(
    cutoff: str = typer.Option(
        None,
        help="YYYY-MM: primer mes de prueba. Default: params.yaml validacion.primer_mes_prueba",
    ),
) -> None:
    """Entrena un Pipeline por objetivo (salidas, llegadas) y lo registra en MLflow."""
    params = load_params()
    if cutoff is None:
        cutoff = params["validacion"]["primer_mes_prueba"]

    peak = params["problema"]["horas_pico"]
    targets = params["problema"]["objetivos"]
    model_cfg = params["modelos"]["hist_gbr"]["fijos"]
    seed = params.get("seed", 42)
    model_cfg_with_seed = {**model_cfg, "random_state": seed}

    logger.info(f"Leyendo panel v3 desde {params['datos']['features']}...")
    df = pd.read_parquet(params["datos"]["features"])
    logger.info(f"Panel: {len(df):,} filas, {df.shape[1]} columnas")

    train, test = temporal_split(df, cutoff)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])

    for target in targets:
        logger.info(f"[{target}] Construyendo Pipeline (ColumnTransformer + HistGBR)...")
        pipeline = build_pipeline(model_cfg_with_seed)

        X_train = train[ALL_FEATURES]
        y_train = train[target].to_numpy(dtype=float)
        X_test = test[ALL_FEATURES]
        y_test = test[target].to_numpy(dtype=float)

        pipeline.fit(X_train, y_train)
        logger.success(f"[{target}] Entrenamiento completo.")

        y_pred = np.clip(pipeline.predict(X_test), 0.0, None)
        metrics = regression_metrics(y_test, y_pred, test["hora"].to_numpy(), peak)
        logger.info(
            f"[{target}] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
            f"MAE-pico={metrics['mae_pico']:.3f}"
        )

        # --- Guardar pipeline ---
        model_path = MODELS_DIR / f"pipeline_{target}.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(pipeline, fh)
        logger.success(f"[{target}] Pipeline guardado en {model_path}")

        # --- Registrar en MLflow ---
        with mlflow.start_run(run_name=f"hist_gbr_{target}"):
            mlflow.set_tags(
                {
                    "tipo": "entrenamiento",
                    "modelo": "HistGradientBoostingRegressor",
                    "objetivo": target,
                    "dataset_version": "v3",
                }
            )
            mlflow.log_params(
                {
                    "objetivo": target,
                    "cutoff_entrenamiento": cutoff,
                    "seed": seed,
                    **{f"model_{key}": value for key, value in model_cfg_with_seed.items()},
                    "n_features": len(ALL_FEATURES),
                    "filas_entrenamiento": len(train),
                    "filas_prueba": len(test),
                }
            )
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    app()
