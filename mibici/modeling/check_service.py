"""Verifica que el servicio desplegado reproduce las predicciones del modelo @champion.

Toma una muestra reproducible de filas reales del panel v3 (mes de referencia de
params.yaml), las predice en esta maquina con el modelo @champion del Model
Registry y las envia una por una al endpoint POST /predict del servicio (por
ejemplo, el contenedor Docker). Compara ambas predicciones fila por fila y el MAE
de cada una contra la demanda real.

Si el servicio corre en otro entorno (contenedor con Python, dependencias y
sistema operativo propios) y las diferencias son cero, el modelo desplegado es
reproducible: produce exactamente los mismos resultados que el de entrenamiento.

Resultados en reports/reproducibilidad/ y en un run de MLflow
("reproducibilidad_servicio").

Uso:
    python -m mibici.modeling.check_service
    python -m mibici.modeling.check_service --url http://localhost:8000 --n 1000
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import requests
import typer

from mibici.config import PROJ_ROOT, load_params
from mibici.modeling.predict import load_champion
from mibici.modeling.train import ALL_FEATURES, BOOL_FEATURES
from mibici.versioning import TRACKING_URI

app = typer.Typer(help="Compara las predicciones del servicio contra el modelo @champion.")

OUT_DIR = PROJ_ROOT / "reports" / "reproducibilidad"
INT_FEATURES = ("hora", "dia_semana", "mes")


def valid_for_api(df: pd.DataFrame) -> pd.Series:
    """Filas que cumplen el esquema de entrada de la API (campos obligatorios y rangos)."""
    required = ["capacidad", "lat", "lon", *INT_FEATURES, *BOOL_FEATURES]
    return (
        df[required].notna().all(axis=1)
        & (df["capacidad"] > 0)
        & df["lat"].between(20.0, 21.0)
        & df["lon"].between(-104.0, -103.0)
    )


def to_payload(row: pd.Series) -> dict:
    """Convierte una fila del panel al JSON de POST /predict (NaN -> null)."""
    payload = {}
    for col in ALL_FEATURES:
        value = row[col]
        if col in BOOL_FEATURES:
            payload[col] = bool(value)
        elif col in INT_FEATURES:
            payload[col] = int(value)
        else:
            value = float(value)
            payload[col] = None if math.isnan(value) else value
    return payload


def service_predictions(
    rows: pd.DataFrame, post: Callable[[dict], dict], targets: list[str]
) -> pd.DataFrame:
    """Envia cada fila al servicio y devuelve sus predicciones por objetivo."""
    out = [post(to_payload(row)) for _, row in rows.iterrows()]
    return pd.DataFrame({t: [r[f"{t}_pred"] for r in out] for t in targets}, index=rows.index)


def compare(
    local: pd.DataFrame, service: pd.DataFrame, actual: pd.DataFrame, targets: list[str]
) -> dict:
    """Diferencias fila por fila y MAE de cada fuente contra la demanda real."""
    summary = {"filas": len(local)}
    for t in targets:
        diff = (local[t] - service[t]).abs()
        summary[f"{t}_dif_max"] = float(diff.max())
        summary[f"{t}_filas_identicas"] = int((diff == 0).sum())
        summary[f"{t}_mae_local"] = float((local[t] - actual[t]).abs().mean())
        summary[f"{t}_mae_servicio"] = float((service[t] - actual[t]).abs().mean())
    summary["reproducible"] = all(summary[f"{t}_dif_max"] < 1e-9 for t in targets)
    return summary


@app.command()
def main(
    url: str = typer.Option("http://localhost:8000", help="URL base del servicio"),
    n: int = typer.Option(1000, min=1, help="Filas de la muestra"),
    image: str = typer.Option("", help="Imagen Docker evaluada (solo para documentar)"),
) -> None:
    """Compara el servicio contra el @champion local en una muestra de filas reales."""
    params = load_params()
    targets = params["problema"]["objetivos"]
    time_col = params["validacion"]["columna_tiempo"]
    month = pd.Timestamp(params["validacion"]["primer_mes_prueba"] + "-01")
    seed = params.get("seed", 42)

    df = pd.read_parquet(params["datos"]["features"], columns=[time_col, *ALL_FEATURES, *targets])
    df = df[(df[time_col] >= month) & (df[time_col] < month + pd.DateOffset(months=1))]
    valid = valid_for_api(df)
    logger.info(f"Mes {month:%Y-%m}: {len(df):,} filas, {int((~valid).sum()):,} fuera del esquema")
    sample = df[valid].sample(n=min(n, int(valid.sum())), random_state=seed)

    local = pd.DataFrame(
        {
            t: np.clip(load_champion(params, t).predict(sample[ALL_FEATURES]), 0, None)
            for t in targets
        },
        index=sample.index,
    )

    session = requests.Session()

    def post(payload: dict) -> dict:
        response = session.post(f"{url}/predict", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    logger.info(f"Enviando {len(sample):,} filas a {url}/predict ...")
    service = service_predictions(sample, post, targets)
    summary = {
        "url": url,
        "imagen": image or "-",
        "mes": f"{month:%Y-%m}",
        "filas_fuera_del_esquema": int((~valid).sum()),
        **compare(local, service, sample[targets], targets),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = sample[[time_col, *targets]].copy()
    for t in targets:
        detail[f"{t}_pred_local"] = local[t]
        detail[f"{t}_pred_servicio"] = service[t]
    detail.to_csv(OUT_DIR / "comparacion_servicio.csv", index=False)
    (OUT_DIR / "resumen_servicio.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])
    with mlflow.start_run(run_name="reproducibilidad_servicio"):
        mlflow.set_tags({"tipo": "reproducibilidad", "imagen": summary["imagen"]})
        mlflow.log_params({"url": url, "filas": summary["filas"], "mes": summary["mes"]})
        mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, int | float)})
        mlflow.log_artifact(str(OUT_DIR / "resumen_servicio.json"))

    for t in targets:
        logger.info(
            f"{t}: diferencia maxima {summary[f'{t}_dif_max']:.2e}, "
            f"{summary[f'{t}_filas_identicas']:,}/{summary['filas']:,} identicas, "
            f"MAE local {summary[f'{t}_mae_local']:.4f} vs servicio "
            f"{summary[f'{t}_mae_servicio']:.4f}"
        )
    if summary["reproducible"]:
        logger.success("Reproducible: el servicio da exactamente las mismas predicciones")
    else:
        logger.warning("El servicio NO reproduce las predicciones locales; revisar dependencias")


if __name__ == "__main__":
    app()
