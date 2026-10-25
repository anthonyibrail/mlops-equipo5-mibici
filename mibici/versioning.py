"""Versionado de datasets con MLflow.

Cada version del dataset (v0 crudo unificado, v1 limpio, v2 features) se registra
como un run del experimento "mibici-datasets" con:
  - tags: version, version padre, etapa
  - params: filas, columnas, rango de fechas, hash SHA-256 del archivo, ruta
  - input dataset (mlflow.log_input) para verla en la columna "Datasets" de la UI
  - artefactos: perfil de columnas y, si existe, el reporte de limpieza

Uso:
    python -m mibici.versioning v0
    python -m mibici.versioning v1 --parent v0
    python -m mibici.versioning v2 --parent v1 --date-col hora_inicio
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
)
EXPERIMENT = "mibici-datasets"

VERSIONS = {
    "v0": (
        "trips_v0.parquet",
        (
            "Crudo unificado: 33 CSV mensuales con esquema, codificacion "
            "y formato de fecha homologados. Sin eliminar filas."
        ),
        None,
    ),
    "v1": (
        "trips_v1.parquet",
        "Limpio: reglas de cleaning.py aplicadas sobre v0.",
        "cleaning_report_v1.json",
    ),
    "v2": (
        "demand_features_v2.parquet",
        (
            "Features: panel (hora_inicio x estacion_id) con features temporales, "
            "de rezago (1h, 24h, 168h), media movil 7d, demanda del sistema y "
            "atributos estaticos de estacion. Generado desde trips_v1 con features.py."
        ),
        None,
    ),
}


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def column_profile(df: pd.DataFrame) -> dict:
    return {
        c: {
            "dtype": str(df[c].dtype),
            "nulos": int(df[c].isna().sum()),
            "unicos": int(df[c].nunique()),
        }
        for c in df.columns
    }


class DatasetVersioner:
    def __init__(self, tracking_uri: str = TRACKING_URI, experiment: str = EXPERIMENT):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_version(
        self,
        df: pd.DataFrame,
        path: Path,
        version: str,
        description: str,
        parent: str | None = None,
        report: Path | None = None,
        date_col: str = "inicio",
    ) -> str:
        """Registra una version del dataset como run de MLflow.

        Args:
            df: DataFrame del dataset a registrar.
            path: Ruta al archivo parquet en disco.
            version: Identificador de la version (ej. 'v0', 'v1', 'v2').
            description: Descripcion del dataset.
            parent: Version padre, si aplica.
            report: Ruta al reporte JSON de limpieza, si aplica.
            date_col: Nombre de la columna de fecha para calcular rango temporal.
                      Por defecto 'inicio' (v0/v1). Usar 'hora_inicio' para v2.
        """
        # Determinar columna de estacion disponible en este dataset
        station_col = "origen_id" if "origen_id" in df.columns else "estacion_id"

        with mlflow.start_run(run_name=f"dataset_{version}") as run:
            mlflow.set_tags(
                {
                    "dataset_version": version,
                    "parent_version": parent or "-",
                    "stage": "data",
                    "mlflow.note.content": description,
                }
            )
            mlflow.log_params(
                {
                    "version": version,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "date_min": str(df[date_col].min()),
                    "date_max": str(df[date_col].max()),
                    "stations": int(df[station_col].nunique()),
                    "sha256": file_sha256(path),
                    "path": path.relative_to(PROJ_ROOT).as_posix(),
                }
            )
            mlflow.log_metric("rows", len(df))
            self._log_input(df, path, version)
            mlflow.log_dict(column_profile(df), f"profile_{version}.json")
            if report and report.exists():
                mlflow.log_artifact(str(report))
                for r in json.loads(report.read_text(encoding="utf-8"))["reglas"]:
                    mlflow.log_metric(f"rule_{r['regla'].split(' ')[0]}", r["filas_afectadas"])
            logger.success(f"{version} registrado en MLflow (run {run.info.run_id})")
            return run.info.run_id

    @staticmethod
    def _log_input(df: pd.DataFrame, path: Path, version: str) -> None:
        """Registra el dataset como mlflow.data input.

        Convierte tipos que MLflow no puede inferir directamente (nullable types,
        tipos de baja precision) a float64/str antes de intentar el registro.
        """
        light = df.copy()
        _non_standard = ("Int64", "boolean", "int8", "int16", "float32")
        for c in light.columns:
            dtype_str = str(light[c].dtype)
            if dtype_str in _non_standard:
                light[c] = light[c].astype("float64")
            elif dtype_str == "category":
                light[c] = light[c].astype(str)
        try:
            ds = mlflow.data.from_pandas(
                light, source=path.as_posix(), name=f"mibici_trips_{version}"
            )
            mlflow.log_input(ds, context=version)
        except Exception as exc:  # noqa: BLE001 - el resto del registro sigue siendo valido
            logger.warning(f"No se pudo registrar el input dataset: {exc}")


@app.command()
def main(
    version: str,
    parent: str = typer.Option(None, help="Version padre, ej. v0"),
    date_col: str = typer.Option("inicio", help="Columna de fecha para rango temporal"),
) -> None:
    """Registra una version del dataset en MLflow.

    Busca el archivo en data/interim y data/processed (en ese orden).
    Ejemplo: python -m mibici.versioning v2 --parent v1 --date-col hora_inicio
    """
    file_name, description, report_name = VERSIONS[version]
    # Resolver ruta: v0/v1 en interim, v2 en processed
    path: Path | None = None
    for data_dir in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR):
        candidate = data_dir / file_name
        if candidate.exists():
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            f"No se encontro {file_name} en data/interim ni data/processed. "
            "Ejecuta primero el paso de generacion correspondiente."
        )
    report = REPORTS_DIR / report_name if report_name else None
    DatasetVersioner().log_version(
        pd.read_parquet(path), path, version, description, parent, report, date_col
    )


if __name__ == "__main__":
    app()
