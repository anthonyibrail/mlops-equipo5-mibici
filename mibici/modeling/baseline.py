"""Experimento con las versiones del dataset: baseline ingenuo estacional.

Para cada version (v0, v1) se construye la demanda horaria de salidas por estacion
y se evalua el pronostico "misma hora de la semana anterior" en los ultimos 3 meses.
El objetivo no es el modelo final (Fase 2), sino medir como cambia el problema al
limpiar los datos y dejar una referencia minima registrada en MLflow.

Uso:
    python -m mibici.modeling.baseline
"""

from __future__ import annotations

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR
from mibici.features import hourly_demand
from mibici.versioning import TRACKING_URI

app = typer.Typer()

EXPERIMENT = "mibici-baseline-versiones"
TEST_MONTHS = 3
SEASON_H = 24 * 7


def seasonal_naive_scores(demand: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    pred = demand.shift(SEASON_H)
    y, yhat = demand[demand.index >= test_start], pred[pred.index >= test_start]
    err = (y - yhat).to_numpy(dtype=float)
    peak = y.index.hour.isin([7, 8, 9, 17, 18, 19])
    mae = float(np.nanmean(np.abs(err)))
    return {
        "mae": mae,
        # MAE / demanda media: compara versiones con distinto nivel de demanda
        "mae_relativo": mae / float(y.to_numpy().mean()),
        "rmse": float(np.sqrt(np.nanmean(err**2))),
        "mae_horas_pico": float(np.nanmean(np.abs(err[peak]))),
        "demanda_media_hora_estacion": float(y.to_numpy().mean()),
    }


@app.command()
def main(versions: list[str] = typer.Option(["v0", "v1"])):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for v in versions:
        df = pd.read_parquet(
            INTERIM_DATA_DIR / f"trips_{v}.parquet", columns=["inicio", "origen_id"]
        ).dropna()
        demand = hourly_demand(df)
        test_start = (
            demand.index.max().normalize()
            - pd.DateOffset(months=TEST_MONTHS)
            + pd.Timedelta(days=1)
        )
        scores = seasonal_naive_scores(demand, test_start)
        with mlflow.start_run(run_name=f"seasonal_naive_{v}"):
            mlflow.set_tags({"dataset_version": v, "model": "seasonal_naive"})
            mlflow.log_params(
                {
                    "dataset_version": v,
                    "season_hours": SEASON_H,
                    "test_start": str(test_start.date()),
                    "stations": demand.shape[1],
                    "trips": len(df),
                }
            )
            mlflow.log_metrics(scores)
        logger.info(f"{v}: {scores}")


if __name__ == "__main__":
    app()
