"""Entrena modelos finales MiBici y los versiona en MLflow Model Registry.

La validacion walk-forward y seleccion de hiperparametros deben estar completas.
El refit utiliza la ventana movil mas reciente de params.yaml (12 meses),
sin afirmar que los meses reutilizados sean un holdout independiente.

Uso desde la raiz del repo:
    python -m mibici.modeling.register
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from mibici.config import load_params
from mibici.modeling.train import ALL_FEATURES, build_pipeline
from mibici.modeling.tune import RESULTS_DIR, WINNERS_PATH, load_feature_panel
from mibici.versioning import TRACKING_URI

app = typer.Typer(help="Refit final y registro versionado de modelos MiBici.")


def training_boundaries(end_month: str, train_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Limites [inicio, fin) de la ventana movil final."""
    if train_months < 1:
        raise ValueError("train_months debe ser mayor a cero")
    try:
        end = pd.Timestamp(end_month + "-01")
    except (ValueError, TypeError) as exc:
        raise ValueError("end_month debe tener formato YYYY-MM") from exc
    if end.strftime("%Y-%m") != end_month:
        raise ValueError("end_month debe tener formato YYYY-MM")
    return end - pd.DateOffset(months=train_months), end


def validate_preds(pred: np.ndarray, max_prediction: float = 1000.0) -> np.ndarray:
    """Falla ante valores imposibles o explosiones numericas no negativas."""
    pred = np.asarray(pred, dtype=float)
    if pred.size == 0 or not np.isfinite(pred).all():
        raise ValueError("Predicciones vacias, infinitas o NaN")
    if np.any(pred < -1e-8):
        raise ValueError("Predicciones negativas para demanda Poisson")
    if float(pred.max()) > max_prediction:
        raise ValueError(f"Prediccion extrema detectada (> {max_prediction:g} bicicletas/estacion/hora)")
    return np.clip(pred, 0, None)


def summarized_validation(report: pd.DataFrame, target: str) -> dict[str, float]:
    """Recupera metricas ya medidas, sin volver a validar ni alterar el reporte."""
    rows = report[report["objetivo"] == target]
    if len(rows) != 6 or not np.isfinite(rows["mae"].to_numpy()).all():
        raise ValueError(f"Informe de seis cortes incompleto o invalido: {target}")
    result = {"mae_seis_cortes": float(rows["mae"].mean())}
    holdout = rows[rows["tipo_corte"] == "holdout"]
    if not holdout.empty:
        result["mae_holdout_diagnostico"] = float(holdout["mae"].mean())
    return result


@app.command()
def main(
    end_month: str = typer.Option(
        "2026-10", help="Primer mes fuera del entrenamiento final; incluye hasta septiembre 2026"
    ),
    max_prediction: float = typer.Option(
        1000.0, min=1, help="Limite de seguridad para predicciones por estacion-hora"
    ),
) -> None:
    """Refit, registra ambos objetivos y comprueba que se puedan recargar."""
    if not WINNERS_PATH.exists():
        raise FileNotFoundError(f"No existe {WINNERS_PATH}; ejecuta el tuning antes")
    report_path = RESULTS_DIR / "walk_forward_6_cortes.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"No existe {report_path}; termina la evaluacion primero")

    params = load_params()
    chosen = json.loads(WINNERS_PATH.read_text(encoding="utf-8"))
    report = pd.read_csv(report_path)
    months = params["validacion"]["meses_entrenamiento"]
    if months is None:
        raise ValueError("Se necesita una ventana de entrenamiento finita para el refit final")
    start, end = training_boundaries(end_month, int(months))
    targets = params["problema"]["objetivos"]
    scores = {t: summarized_validation(report, t) for t in targets}

    df = load_feature_panel(params)
    time_col = params["validacion"]["columna_tiempo"]
    training = df[(df[time_col] >= start) & (df[time_col] < end)]
    if training.empty:
        raise ValueError(f"Sin datos entre {start.date()} y {end.date()}")
    if pd.Timestamp(training[time_col].max()) < end - pd.Timedelta(days=2):
        raise ValueError("El periodo mas reciente parece incompleto; verificar archivos")

    logger.info(f"Refit final: {len(training):,} filas entre {start.date()} y {end.date()} (exclusivo)")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])
    client = MlflowClient()
    base_name = params["mlflow"]["modelo_registrado"]
    seed = params.get("seed", 42)

    for target in targets:
        config = {**chosen["winners"][target]["config"], "random_state": seed}
        pipeline = build_pipeline(config)
        logger.info(f"Entrenamiento final de {target}...")
        X = training[ALL_FEATURES]
        y = training[target]
        if y.isna().any() or (y < 0).any():
            raise ValueError(f"Objetivo invalido para Poisson: {target}")
        pipeline.fit(X, y)

        # Diagnostico reproducible en filas recientes del ENTRENAMIENTO; no es test independiente.
        latest = training[training[time_col] >= end - pd.DateOffset(months=1)]
        sample = latest.sample(n=min(3000, len(latest)), random_state=seed)
        predictions = validate_preds(pipeline.predict(sample[ALL_FEATURES]), max_prediction)
        example = sample[ALL_FEATURES].head(5)
        example_pred = validate_preds(pipeline.predict(example), max_prediction)
        signature = infer_signature(example, example_pred)
        registered_name = f"{base_name}-{target}"

        with mlflow.start_run(run_name=f"registro_final_{target}") as run:
            mlflow.set_tags({
                "tipo": "modelo_final_registrado",
                "objetivo": target,
                "dataset_version": "v3",
                "estado": "candidate",
                "advertencia_evaluacion": "Junio 2026 se consulto para corregir inestabilidad: holdout no totalmente independiente",
            })
            mlflow.log_params({
                "target": target, "training_from": str(start.date()),
                "training_until_exclusive": str(end.date()),
                "rows_train": len(training), "seed": seed,
                **{f"model_{k}": v for k, v in config.items()},
            })
            mlflow.log_metrics({
                **scores[target],
                "max_pred_inspeccion_entrenamiento": float(predictions.max()),
            })
            mlflow.log_dict(chosen["winners"][target], f"seleccion_{target}.json")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=f"pipeline_{target}",
                signature=signature,
                input_example=example,
                serialization_format="cloudpickle",
                registered_model_name=registered_name,
            )
            current_run_id = run.info.run_id

        versions = [
            v for v in client.search_model_versions(f"name = '{registered_name}'")
            if v.run_id == current_run_id
        ]
        if not versions:
            raise RuntimeError(f"Modelo registrado, pero no se encontro version de {registered_name}; revisar MLflow")
        version = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(registered_name, "candidate", version.version)
        client.set_model_version_tag(
            registered_name, version.version, "validacion", "walk_forward_6_cortes"
        )
        reloaded = mlflow.sklearn.load_model(f"models:/{registered_name}/{version.version}")
        reproduced = validate_preds(reloaded.predict(example), max_prediction)
        if not np.allclose(reproduced, example_pred, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"Predicciones distintas tras recargar version {version.version}")
        logger.success(
            f"REGISTRADO {registered_name} v{version.version} (@candidate)"
            f" | MAE 6 cortes: {scores[target]['mae_seis_cortes']:.4f}"
        )
        del pipeline, reloaded, X, y
        gc.collect()

    logger.success("Registro finalizado; consulta Model Registry en la base local MLflow")


if __name__ == "__main__":
    app()
