"""Ajuste de hiperparametros con RandomizedSearchCV (Fase 2 - en progreso).

Ejecuta RandomizedSearchCV sobre el modelo indicado usando WalkForwardSplit
como esquema de validacion cruzada temporal. El mejor estimador se evalua
sobre el conjunto de prueba y se registra en MLflow.

Estado actual (en progreso):
  - hist_gbr: busqueda implementada para 'salidas'. Falta: 'llegadas',
    comparacion entre modelos y registro en Model Registry.
  - random_forest: estructura definida en params.yaml; entrenamiento pendiente.
  - TODO: completar multi-modelo y registrar el ganador en MLflow Model Registry.

Uso:
    python -m mibici.modeling.tune                       # hist_gbr / salidas
    python -m mibici.modeling.tune --modelo hist_gbr --objetivo salidas
"""

from __future__ import annotations

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import typer

from mibici.config import load_params
from mibici.modeling import bool_to_int  # noqa: F401 (importado para compatibilidad de pickle)
from mibici.modeling.train import ALL_FEATURES, build_pipeline, temporal_split
from mibici.modeling.validation import WalkForwardSplit, regression_metrics
from mibici.versioning import TRACKING_URI

app = typer.Typer()


def build_estimator_for_search(nombre_modelo: str, params: dict):
    """Devuelve el estimador base para RandomizedSearchCV segun el modelo elegido.

    Args:
        nombre_modelo: Clave del modelo en params.yaml (p. ej. 'hist_gbr').
        params:        Resultado de load_params().

    Returns:
        Estimador de sklearn sin entrenar.
    """
    seed = params.get("seed", 42)
    if nombre_modelo == "hist_gbr":
        cfg = params["modelos"]["hist_gbr"]["fijos"]
        return HistGradientBoostingRegressor(**cfg, random_state=seed)
    if nombre_modelo == "random_forest":
        cfg = params["modelos"]["random_forest"]["fijos"]
        # TODO Data Scientist: RandomForest no tiene loss=poisson;
        #      considerar transformar el objetivo (log1p) o usar otros criterios.
        return RandomForestRegressor(**cfg, random_state=seed)
    raise ValueError(f"Modelo desconocido: '{nombre_modelo}'. Opciones: hist_gbr, random_forest")


def param_grid_from_yaml(nombre_modelo: str, params: dict) -> dict:
    """Convierte el espacio de busqueda de params.yaml al formato de sklearn.

    RandomizedSearchCV espera listas de valores (no dicts anidados).

    Args:
        nombre_modelo: Clave del modelo en params.yaml.
        params:        Resultado de load_params().

    Returns:
        Diccionario {param_name: [values]} con el prefijo 'model__'.
    """
    espacio = params["modelos"][nombre_modelo]["busqueda"]["espacio"]
    return {f"model__{k}": v for k, v in espacio.items()}


@app.command()
def main(
    modelo: str = typer.Option("hist_gbr", help="Modelo a ajustar: hist_gbr | random_forest"),
    objetivo: str = typer.Option("salidas", help="Variable objetivo: salidas | llegadas"),
) -> None:
    """Ajusta hiperparametros de un modelo con RandomizedSearchCV y WalkForwardSplit."""
    params = load_params()
    peak = params["problema"]["horas_pico"]
    busqueda_cfg = params["modelos"][modelo]["busqueda"]

    # --- Solo hist_gbr/salidas esta completamente implementado ---
    if modelo == "random_forest":
        # TODO Data Scientist: implementar Pipeline para RandomForest.
        #      RandomForest no maneja NaN nativamente; agregar SimpleImputer
        #      al ColumnTransformer antes de continuar.
        logger.warning(
            "RandomForest requiere un SimpleImputer para NaN en los lags. "
            "Implementacion pendiente. Saliendo."
        )
        raise typer.Exit(code=1)

    logger.info(f"Ajustando '{modelo}' para objetivo '{objetivo}'...")

    df = pd.read_parquet(params["datos"]["features"])
    cutoff = params["validacion"]["primer_mes_prueba"]
    train, test = temporal_split(df, cutoff)

    # --- Pipeline base sin hiperparametros fijos (para que la busqueda los sobreescriba) ---
    base_cfg = {k: v for k, v in params["modelos"][modelo]["fijos"].items()
                if k not in busqueda_cfg["espacio"]}
    base_cfg["random_state"] = params.get("seed", 42)
    pipeline = build_pipeline(base_cfg)

    # --- WalkForwardSplit como CV temporal ---
    # NOTA: se usa solo 1 corte (el primero) para acelerar la busqueda.
    # TODO ML Engineer: cambiar n_cortes a params["validacion"]["n_cortes"]
    #      para usar todos los cortes antes de registrar el modelo ganador.
    splitter_1fold = WalkForwardSplit(
        time_col=params["validacion"]["columna_tiempo"],
        first_test_month=params["validacion"]["primer_mes_prueba"],
        n_splits=1,
        test_months=params["validacion"]["meses_prueba"],
        train_months=params["validacion"]["meses_entrenamiento"],
    )

    param_grid = param_grid_from_yaml(modelo, params)
    n_iter = busqueda_cfg["n_iter"]

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=list(splitter_1fold.split(train)),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        random_state=params.get("seed", 42),
        verbose=1,
        refit=True,
    )

    logger.info(f"Ejecutando RandomizedSearchCV ({n_iter} iteraciones, 1 corte)...")
    X_train = train[ALL_FEATURES]
    y_train = train[objetivo].to_numpy(dtype=float)
    search.fit(X_train, y_train)

    logger.success(f"Mejores parametros: {search.best_params_}")
    logger.info(f"Mejor MAE CV (negativo): {search.best_score_:.4f}")

    # --- Evaluacion sobre el conjunto de prueba ---
    best_pipe = search.best_estimator_
    y_pred = np.clip(best_pipe.predict(test[ALL_FEATURES]), 0.0, None)
    metrics = regression_metrics(test[objetivo].to_numpy(float), y_pred, test["hora"].to_numpy(), peak)
    logger.info(
        f"Test -> MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
        f"MAE-pico={metrics['mae_pico']:.3f}"
    )

    # --- Registro en MLflow ---
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])

    with mlflow.start_run(run_name=f"tune_{modelo}_{objetivo}"):
        mlflow.set_tags(
            {"tipo": "busqueda", "modelo": modelo, "objetivo": objetivo, "dataset_version": "v3"}
        )
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_param("n_iter_busqueda", n_iter)
        mlflow.log_param("n_cortes_cv", 1)  # TODO: cambiar a n_cortes completo
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # TODO ML Engineer: registrar best_pipe en MLflow Model Registry cuando
        #      la busqueda use todos los cortes y se compare contra las lineas base.
        # mlflow.sklearn.log_model(best_pipe, "pipeline", registered_model_name=...)
        logger.warning(
            "Registro en Model Registry pendiente: completar busqueda con todos los cortes."
        )


if __name__ == "__main__":
    app()
