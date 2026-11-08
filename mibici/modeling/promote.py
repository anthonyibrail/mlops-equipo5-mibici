"""Promocion a @champion de los modelos registrados que superan a la linea base.

Regla (params.yaml > promocion):
    Las versiones @candidate de los modelos de salidas y llegadas reciben el alias
    @champion solo si su MAE promedio en los seis cortes walk-forward es menor que el
    de la linea base de referencia en TODOS los criterios (salidas, llegadas y balance
    neto). La decision es conjunta porque el balance neto depende de ambos modelos.

Fuentes de datos:
    - Modelo:     reports/model_selection/walk_forward_6_cortes.csv (tune evaluate).
    - Linea base: ultimo run de MLflow con tags tipo=linea_base y modelo=<referencia>
                  (make lineas-base).

La decision, la comparacion y las versiones evaluadas quedan en un run de MLflow
("promocion_champion") y como tags de cada version del Model Registry.

Uso:
    python -m mibici.modeling.promote
"""

from __future__ import annotations

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import typer

from mibici.config import load_params
from mibici.modeling.tune import RESULTS_DIR
from mibici.versioning import TRACKING_URI

app = typer.Typer(help="Promocion a @champion de los modelos que superan a la linea base.")

REPORT_PATH = RESULTS_DIR / "walk_forward_6_cortes.csv"


def model_scores(report: pd.DataFrame, criteria: list[str]) -> dict[str, float]:
    """MAE promedio de los cortes walk-forward para cada criterio.

    Args:
        report:   Reporte de tune evaluate, con columnas 'objetivo' y 'mae'.
        criteria: Criterios a comparar (por ejemplo, salidas, llegadas y balance).

    Returns:
        Diccionario criterio -> MAE promedio del modelo.
    """
    scores = {}
    for criterion in criteria:
        rows = report[report["objetivo"] == criterion]
        if rows.empty:
            raise ValueError(f"El reporte no tiene resultados para '{criterion}'")
        scores[criterion] = float(rows["mae"].mean())
    return scores


def baseline_metric_name(criterion: str) -> str:
    """Nombre de la metrica de MLflow que guarda el MAE de la linea base."""
    return "balance_mae" if criterion == "balance" else f"{criterion}_mae_promedio"


def baseline_scores(
    client: MlflowClient, experiment_name: str, baseline: str, criteria: list[str]
) -> tuple[dict[str, float], str]:
    """MAE de la linea base, leido del run mas reciente registrado en MLflow.

    Returns:
        (diccionario criterio -> MAE de la linea base, run_id del que se leyo)
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No existe el experimento '{experiment_name}'; ejecuta make lineas-base")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.tipo = 'linea_base' and tags.modelo = '{baseline}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No hay runs de la linea base '{baseline}'; ejecuta make lineas-base")
    run = runs[0]
    scores = {}
    for criterion in criteria:
        name = baseline_metric_name(criterion)
        if name not in run.data.metrics:
            raise ValueError(f"El run {run.info.run_id} no tiene la metrica '{name}'")
        scores[criterion] = float(run.data.metrics[name])
    return scores, run.info.run_id


def decide_promotion(
    model: dict[str, float], baseline: dict[str, float]
) -> tuple[bool, pd.DataFrame]:
    """Compara modelo contra linea base. Promueve solo si mejora en todos los criterios.

    Returns:
        (decision, tabla con el MAE de ambos, la mejora porcentual y si supera)
    """
    rows = []
    for criterion, base_mae in baseline.items():
        model_mae = model[criterion]
        rows.append(
            {
                "criterio": criterion,
                "mae_modelo": model_mae,
                "mae_linea_base": base_mae,
                "mejora_pct": 100 * (base_mae - model_mae) / base_mae,
                "supera": model_mae < base_mae,
            }
        )
    table = pd.DataFrame(rows)
    return bool(table["supera"].all()), table


@app.command()
def main() -> None:
    """Compara el @candidate contra la linea base y, si la supera, le asigna @champion."""
    params = load_params()
    cfg = params["promocion"]
    criteria = cfg["criterios"]
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"No existe {REPORT_PATH}; ejecuta make evaluate")
    model = model_scores(pd.read_csv(REPORT_PATH), criteria)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = params["mlflow"]["experimento_modelos"]
    baseline, baseline_run = baseline_scores(
        client, experiment, cfg["linea_base_referencia"], criteria
    )
    promote, table = decide_promotion(model, baseline)
    logger.info("Comparacion contra la linea base:\n" + table.to_string(index=False))

    base_name = params["mlflow"]["modelo_registrado"]
    versions = {
        target: client.get_model_version_by_alias(f"{base_name}-{target}", cfg["alias_candidato"])
        for target in params["problema"]["objetivos"]
    }

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="promocion_champion"):
        mlflow.set_tags(
            {
                "tipo": "promocion",
                "decision": "champion" if promote else "rechazada",
                "dataset_version": "v3",
            }
        )
        mlflow.log_params(
            {
                "linea_base_referencia": cfg["linea_base_referencia"],
                "run_linea_base": baseline_run,
                "criterios": ",".join(criteria),
                **{f"version_{t}": v.version for t, v in versions.items()},
            }
        )
        for row in table.itertuples():
            mlflow.log_metrics(
                {
                    f"{row.criterio}_mae_modelo": row.mae_modelo,
                    f"{row.criterio}_mae_linea_base": row.mae_linea_base,
                    f"{row.criterio}_mejora_pct": row.mejora_pct,
                }
            )
        mlflow.log_table(table, "comparacion_linea_base.json")

    for target, version in versions.items():
        name = f"{base_name}-{target}"
        for row in table.itertuples():
            client.set_model_version_tag(
                name, version.version, f"mejora_pct_{row.criterio}", f"{row.mejora_pct:.2f}"
            )
        client.set_model_version_tag(
            name, version.version, "promocion", "champion" if promote else "rechazada"
        )
        if promote:
            client.set_registered_model_alias(name, cfg["alias_produccion"], version.version)
            logger.success(f"{name} v{version.version} -> @{cfg['alias_produccion']}")

    if not promote:
        failed = ", ".join(table.loc[~table["supera"], "criterio"])
        logger.warning(
            f"No se asigna @{cfg['alias_produccion']}: no supera a la linea base en {failed}"
        )


if __name__ == "__main__":
    app()
