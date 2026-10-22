"""Versionado de datasets con MLflow.

Cada version del dataset (v0 crudo unificado, v1 limpio, ...) se registra como un
run del experimento "mibici-datasets" con:
  - tags: version, version padre, etapa
  - params: filas, columnas, rango de fechas, hash SHA-256 del archivo, ruta
  - input dataset (mlflow.log_input) para verla en la columna "Datasets" de la UI
  - artefactos: perfil de columnas y, si existe, el reporte de limpieza

Uso:
    python -m mibici.versioning v0
    python -m mibici.versioning v1 --parent v0
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import typer
from loguru import logger

from mibici.config import INTERIM_DATA_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}")
EXPERIMENT = "mibici-datasets"

VERSIONS = {
    "v0": ("trips_v0.parquet", "Crudo unificado: 33 CSV mensuales con esquema, codificacion "
                               "y formato de fecha homologados. Sin eliminar filas.", None),
    "v1": ("trips_v1.parquet", "Limpio: reglas de cleaning.py aplicadas sobre v0.",
           "cleaning_report_v1.json"),
}


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def column_profile(df: pd.DataFrame) -> dict:
    return {c: {"dtype": str(df[c].dtype), "nulos": int(df[c].isna().sum()),
                "unicos": int(df[c].nunique())} for c in df.columns}


class DatasetVersioner:
    def __init__(self, tracking_uri: str = TRACKING_URI, experiment: str = EXPERIMENT):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_version(self, df: pd.DataFrame, path: Path, version: str, description: str,
                    parent: str | None = None, report: Path | None = None) -> str:
        with mlflow.start_run(run_name=f"dataset_{version}") as run:
            mlflow.set_tags({"dataset_version": version, "parent_version": parent or "-",
                             "stage": "data", "mlflow.note.content": description})
            mlflow.log_params({
                "version": version,
                "rows": len(df),
                "columns": df.shape[1],
                "date_min": str(df["inicio"].min()),
                "date_max": str(df["inicio"].max()),
                "stations": int(df["origen_id"].nunique()),
                "sha256": file_sha256(path),
                "path": path.relative_to(PROJ_ROOT).as_posix(),
            })
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
        # MLflow infiere el esquema a partir de tipos numpy estandar: se convierte una copia.
        light = df.copy()
        for c in light.columns:
            if str(light[c].dtype) in ("Int64", "boolean"):
                light[c] = light[c].astype("float64")
            elif str(light[c].dtype) == "category":
                light[c] = light[c].astype(str)
        try:
            ds = mlflow.data.from_pandas(light, source=path.as_posix(), name=f"mibici_trips_{version}")
            mlflow.log_input(ds, context=version)
        except Exception as exc:  # el resto del registro sigue siendo valido
            logger.warning(f"No se pudo registrar el input dataset: {exc}")


@app.command()
def main(version: str, parent: str = typer.Option(None)):
    file_name, description, report_name = VERSIONS[version]
    path = INTERIM_DATA_DIR / file_name
    report = REPORTS_DIR / report_name if report_name else None
    DatasetVersioner().log_version(pd.read_parquet(path), path, version, description, parent, report)


if __name__ == "__main__":
    app()
