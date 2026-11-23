"""Exporta los modelos @champion del Model Registry para empaquetarlos en Docker.

El Model Registry vive en mlflow.db, con rutas absolutas de la maquina donde se
entreno, asi que no se puede copiar tal cual a un contenedor. Este modulo carga
la version @champion de cada objetivo y la guarda como un archivo independiente
en models/champion/, junto con model_info.json, que documenta exactamente que
version del Registry se empaqueto (nombre, version, run de origen y URI).

El Dockerfile copia models/champion/ a models/ dentro de la imagen, que es donde
el servicio FastAPI (mibici/api/loader.py) busca los pipelines.

Uso:
    python -m mibici.modeling.export
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import pickle

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import typer

from mibici.config import MODELS_DIR, load_params
from mibici.modeling.predict import load_champion
from mibici.versioning import TRACKING_URI

app = typer.Typer(help="Exporta los modelos @champion para el contenedor del servicio.")

EXPORT_DIR = MODELS_DIR / "champion"


def model_info(params: dict, client: MlflowClient, target: str) -> dict:
    """Metadatos de la version @champion de un objetivo en el Model Registry."""
    name = f"{params['mlflow']['modelo_registrado']}-{target}"
    alias = params["promocion"]["alias_produccion"]
    version = client.get_model_version_by_alias(name, alias)
    return {
        "nombre": name,
        "version": int(version.version),
        "alias": alias,
        "uri": f"models:/{name}/{version.version}",
        "run_id": version.run_id,
    }


def export_champions(params: dict, out_dir: Path = EXPORT_DIR) -> dict:
    """Guarda el pipeline @champion de cada objetivo y su model_info.json.

    Returns:
        Contenido de model_info.json.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = {"exportado": datetime.now(UTC).isoformat(timespec="seconds"), "modelos": {}}
    for target in params["problema"]["objetivos"]:
        meta = model_info(params, client, target)
        pipeline = load_champion(params, target)
        path = out_dir / f"pipeline_{target}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(pipeline, fh)
        info["modelos"][target] = {**meta, "archivo": f"pipeline_{target}.pkl"}
        logger.success(f"{meta['uri']} (@{meta['alias']}) -> {path}")

    (out_dir / "model_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info


@app.command()
def main() -> None:
    """Exporta los modelos @champion a models/champion/."""
    export_champions(load_params())
    logger.success(f"Modelos listos para Docker en {EXPORT_DIR}")


if __name__ == "__main__":
    app()
