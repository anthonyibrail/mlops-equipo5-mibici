"""Búsqueda reproducible y evaluación temporal de modelos de demanda MiBici.

Búsqueda: muestra de los dos primeros cortes temporales (2026-04, 2026-05).
Evaluación: seis cortes completos, con modelos elegidos sin usar cortes 2–5
(2026-06 a 2026-09) durante la selección.

Uso:
    python -m mibici.modeling.tune search --n-iter 3
    python -m mibici.modeling.tune evaluate
"""

from __future__ import annotations

import gc
import json

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
import typer

from mibici.config import PROJ_ROOT, load_params
from mibici.modeling.train import ALL_FEATURES, NUMERIC_FEATURES, build_pipeline
from mibici.modeling.validation import (
    WalkForwardSplit,
    net_balance_metrics,
    regression_metrics,
)
from mibici.versioning import TRACKING_URI

app = typer.Typer(help="Hyperparameter tuning y walk-forward en MiBici.")
RESULTS_DIR = PROJ_ROOT / "reports" / "model_selection"
WINNERS_PATH = RESULTS_DIR / "tuning_winners.json"


def load_feature_panel(params: dict) -> pd.DataFrame:
    """Lee únicamente columnas necesarias, optimizando RAM para Mac de 8 GB."""
    cols = list(
        dict.fromkeys(
            [params["validacion"]["columna_tiempo"]]
            + ALL_FEATURES
            + list(params["problema"]["objetivos"])
        )
    )
    df = pd.read_parquet(params["datos"]["features"], columns=cols)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for target in params["problema"]["objetivos"]:
        df[target] = pd.to_numeric(df[target], errors="coerce").astype("float32")
    if df[params["validacion"]["columna_tiempo"]].isna().any():
        raise ValueError("La columna temporal contiene fechas nulas")
    logger.info(f"Panel para modelado: {len(df):,} filas, {df.shape[1]} columnas")
    return df


def parameter_candidates(params: dict, n_iter: int) -> list[dict]:
    """Incluye siempre la configuración actual para evitar elegir algo peor."""
    if n_iter < 1:
        raise ValueError("n_iter debe ser >= 1")
    model = params["modelos"]["hist_gbr"]
    fixed = dict(model["fijos"])
    sampled = ParameterSampler(
        model["busqueda"]["espacio"],
        n_iter=n_iter,
        random_state=params.get("seed", 42),
    )
    candidates = [fixed]
    for changes in sampled:
        candidate = {**fixed, **changes}
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def evaluate_sampled_fold(
    df: pd.DataFrame,
    target: str,
    train_index: np.ndarray,
    test_index: np.ndarray,
    config: dict,
    peak_hours: list[int],
    seed: int,
) -> dict[str, float]:
    model = build_pipeline({**config, "random_state": seed})
    train = df.iloc[train_index]
    test = df.iloc[test_index]
    model.fit(train[ALL_FEATURES], train[target])
    pred = model.predict(test[ALL_FEATURES])
    if not np.isfinite(pred).all():
        raise ValueError("Predicciones no finitas para esta configuración")
    pred = np.clip(pred, 0.0, None)
    if not np.isfinite(pred).all():
        raise ValueError("Predicciones inválidas para esta configuración")
    return regression_metrics(test[target], pred, test["hora"], peak_hours)


def sample_index(indices: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows < 1:
        raise ValueError("max_rows debe ser positivo")
    if len(indices) <= max_rows:
        return indices
    return np.sort(rng.choice(indices, size=max_rows, replace=False))


@app.command()
def search(
    n_iter: int = typer.Option(
        3, min=1, help="Configuraciones aleatorias adicionales a la original"
    ),
    train_rows: int = typer.Option(
        180_000, min=1, help="Filas máximas de entrenamiento por corte"
    ),
    test_rows: int = typer.Option(60_000, min=1, help="Filas máximas de prueba por corte"),
    tune_folds: int = typer.Option(
        2, min=1, max=5, help="Primeros cortes usados para selección (el resto queda como holdout)"
    ),
) -> None:
    """Busca parámetros con muestra reproducible y sin tocar los modelos existentes."""
    params = load_params()
    seed = params.get("seed", 42)
    splitter = WalkForwardSplit.from_params(params)
    if tune_folds >= splitter.n_splits:
        raise typer.BadParameter("Deja por lo menos un corte sin usar para tuning")
    df = load_feature_panel(params)
    rng = np.random.default_rng(seed)
    sampled_folds = []
    for number, (tr, te) in enumerate(splitter.split(df)):
        if number >= tune_folds:
            break
        sampled_folds.append(
            (
                sample_index(tr, train_rows, rng),
                sample_index(te, test_rows, rng),
            )
        )
    candidates = parameter_candidates(params, n_iter)
    peak = params["problema"]["horas_pico"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])
    results = []
    winners = {}
    for target in params["problema"]["objetivos"]:
        for idx, config in enumerate(candidates):
            logger.info(f"Tuning {target}: candidato {idx + 1}/{len(candidates)}")
            fold_metrics = []
            error = None
            for fold_number, (tr, te) in enumerate(sampled_folds):
                try:
                    score = evaluate_sampled_fold(df, target, tr, te, config, peak, seed)
                    fold_metrics.append(score)
                    logger.info(f"  corte {fold_number}: MAE={score['mae']:.4f}")
                except (FloatingPointError, OverflowError, ValueError) as exc:
                    error = str(exc)
                    logger.warning(f"Configuración descartada: {error}")
                    break
                finally:
                    gc.collect()
            mean_mae = (
                float(np.mean([f["mae"] for f in fold_metrics]))
                if (error is None and len(fold_metrics) == tune_folds)
                else float("inf")
            )
            row = {
                "objetivo": target,
                "candidato": idx,
                "mae_seleccion": mean_mae,
                "estado": "valido" if np.isfinite(mean_mae) else "descartado",
                "error": error or "",
                **{f"param_{k}": v for k, v in config.items()},
            }
            results.append(row)
            with mlflow.start_run(run_name=f"tune_{target}_{idx}"):
                mlflow.set_tags(
                    {
                        "tipo": "tuning_muestreado",
                        "objetivo": target,
                        "dataset_version": "v3",
                        "cortes_tuning": str(tune_folds),
                    }
                )
                mlflow.log_params(
                    {
                        **{f"model_{k}": v for k, v in config.items()},
                        "n_train_max": train_rows,
                        "n_test_max": test_rows,
                        "n_cortes_tuning": tune_folds,
                        "semilla": seed,
                    }
                )
                if np.isfinite(mean_mae):
                    mlflow.log_metric("mae_seleccion", mean_mae)
                    for k, metric in enumerate(fold_metrics):
                        mlflow.log_metric("mae_corte", metric["mae"], step=k)
                if error:
                    mlflow.set_tag("error", error[:500])
            logger.info(f"  MAE medio de selección: {mean_mae:.4f}")
        valid = [r for r in results if r["objetivo"] == target and r["estado"] == "valido"]
        if not valid:
            raise RuntimeError(f"Ningún candidato válido para {target}")
        best = min(valid, key=lambda r: r["mae_seleccion"])
        winners[target] = {
            "mae_seleccion": best["mae_seleccion"],
            "config": candidates[best["candidato"]],
            "candidato": best["candidato"],
        }
        logger.success(f"Mejor {target}: MAE muestral={best['mae_seleccion']:.4f}")
    report = pd.DataFrame(results)
    report.to_csv(RESULTS_DIR / "tuning_search.csv", index=False)
    WINNERS_PATH.write_text(
        json.dumps(
            {
                "seed": seed,
                "metodologia": "primeros cortes temporales para tuning; muestra aleatoria reproducible por corte",
                "tune_folds": tune_folds,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "winners": winners,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.success(f"Resultados: {RESULTS_DIR / 'tuning_search.csv'}")
    logger.success(f"Ganadores: {WINNERS_PATH}")


@app.command()
def evaluate() -> None:
    """Verifica ganadores con seis cortes completos; registra resultados en MLflow.

    ATENCIÓN: los primeros tune_folds también se usaron durante selección.
    Reportar además el desempeño holdout en los cortes posteriores.
    """
    if not WINNERS_PATH.exists():
        raise FileNotFoundError(f"Primero ejecuta search: falta {WINNERS_PATH}")
    chosen = json.loads(WINNERS_PATH.read_text(encoding="utf-8"))
    params = load_params()
    splitter = WalkForwardSplit.from_params(params)
    df = load_feature_panel(params)
    peak = params["problema"]["horas_pico"]
    targets = params["problema"]["objetivos"]
    seed = params.get("seed", 42)
    folds = splitter.folds()
    tuning_cut = chosen["tune_folds"]
    rows = []
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])
    with mlflow.start_run(run_name="hist_gbr_walk_forward_6_cortes"):
        mlflow.set_tags(
            {
                "tipo": "evaluacion_walk_forward",
                "dataset_version": "v3",
                "advertencia": "Los primeros cortes se usaron para seleccionar hiperparametros; consultar holdout",
            }
        )
        mlflow.log_params(
            {
                "n_cortes": splitter.n_splits,
                "cortes_tuning": tuning_cut,
                "meses_entrenamiento": splitter.train_months,
                "semilla": seed,
            }
        )
        for k, (tr, te) in enumerate(splitter.split(df)):
            pred_by_target = {}
            logger.info(
                f"Validación {k + 1}/{splitter.n_splits}: {folds.loc[k, 'inicio_prueba']:%Y-%m}"
            )
            test = df.iloc[te]
            for target in targets:
                config = chosen["winners"][target]["config"]
                pipeline = build_pipeline({**config, "random_state": seed})
                train = df.iloc[tr]
                pipeline.fit(train[ALL_FEATURES], train[target])
                del train
                pred = pipeline.predict(test[ALL_FEATURES])
                if not np.isfinite(pred).all():
                    raise ValueError(f"Predicciones no finitas: {target}, corte {k}")
                pred = np.clip(pred, 0.0, None)
                pred_by_target[target] = pred
                metric = regression_metrics(test[target], pred, test["hora"], peak)
                rows.append(
                    {
                        "corte": k,
                        "mes_prueba": folds.loc[k, "inicio_prueba"].strftime("%Y-%m"),
                        "objetivo": target,
                        "tipo_corte": "seleccion" if k < tuning_cut else "holdout",
                        "filas_entrenamiento": len(tr),
                        "filas_prueba": len(te),
                        **metric,
                    }
                )
                mlflow.log_metric(f"{target}_mae_corte", metric["mae"], step=k)
                logger.success(f"  {target}: MAE={metric['mae']:.4f}")
                del pipeline
                gc.collect()
            if {"salidas", "llegadas"}.issubset(pred_by_target):
                balance = net_balance_metrics(
                    test["salidas"],
                    test["llegadas"],
                    pred_by_target["salidas"],
                    pred_by_target["llegadas"],
                    test["hora"],
                    peak,
                )
                rows.append(
                    {
                        "corte": k,
                        "mes_prueba": folds.loc[k, "inicio_prueba"].strftime("%Y-%m"),
                        "objetivo": "balance",
                        "tipo_corte": "seleccion" if k < tuning_cut else "holdout",
                        "filas_entrenamiento": len(tr),
                        "filas_prueba": len(te),
                        "mae": balance["balance_mae"],
                        "rmse": balance["balance_rmse"],
                        "mae_pico": balance["balance_mae_pico"],
                    }
                )
                mlflow.log_metric("balance_mae_corte", balance["balance_mae"], step=k)
                logger.success(f"  balance: MAE={balance['balance_mae']:.4f}")
            del test, pred_by_target
            gc.collect()
        report = pd.DataFrame(rows)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "walk_forward_6_cortes.csv"
        report.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
        for target in list(targets) + (
            ["balance"] if "balance" in report["objetivo"].values else []
        ):
            relevant = report[report["objetivo"] == target]
            for suffix, subset in (
                ("seis_cortes", relevant),
                ("holdout", relevant[relevant["tipo_corte"] == "holdout"]),
            ):
                for field in ["mae", "rmse", "mae_pico"]:
                    value = float(subset[field].mean())
                    mlflow.log_metric(f"{target}_{field}_{suffix}", value)
                logger.success(f"{target} {suffix}: MAE={subset['mae'].mean():.4f}")
        for target in targets:
            mlflow.log_dict(chosen["winners"][target]["config"], f"parametros_{target}.json")
    logger.success(f"Reporte final: {path}")


if __name__ == "__main__":
    app()
