"""Pruebas de la exportacion de los modelos @champion para el contenedor.

Usan un MLflow temporal con un Pipeline real de train.py entrenado con pocas filas
sinteticas: comprueban que el archivo exportado conserva las predicciones del
modelo del Registry y que el servicio FastAPI lo carga y responde con el.
"""

import json

from fastapi.testclient import TestClient
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici.api import loader
from mibici.modeling import export, predict
from mibici.modeling.train import ALL_FEATURES, BOOL_FEATURES, build_pipeline

PARAMS = {
    "problema": {"objetivos": ["salidas", "llegadas"]},
    "mlflow": {"modelo_registrado": "prueba-demanda"},
    "promocion": {"alias_produccion": "champion"},
}


def _features(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.random(n) * 5 for c in ALL_FEATURES})
    for c in BOOL_FEATURES:
        X[c] = rng.random(n) > 0.5
    return X


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Registry temporal con un @champion por objetivo (version 2 de cada uno)."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(export, "TRACKING_URI", uri)
    monkeypatch.setattr(predict, "TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    X = _features(200)
    for target, scale in (("salidas", 1.0), ("llegadas", 2.0)):
        name = f"prueba-demanda-{target}"
        for _ in range(2):  # dos versiones; champion = la segunda
            pipe = build_pipeline({"loss": "poisson", "max_iter": 20, "random_state": 42})
            pipe.fit(X, scale * X["hora"] + 1)
            with mlflow.start_run():
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name="modelo",
                    registered_model_name=name,
                    serialization_format="cloudpickle",  # igual que register.py
                )
        client.set_registered_model_alias(name, "champion", "2")
    return tmp_path


def test_exporta_champion_con_sus_metadatos(registry):
    out = registry / "champion"
    info = export.export_champions(PARAMS, out)

    meta = json.loads((out / "model_info.json").read_text(encoding="utf-8"))
    assert meta == info
    for target in ("salidas", "llegadas"):
        assert (out / f"pipeline_{target}.pkl").exists()
        assert meta["modelos"][target]["version"] == 2
        assert meta["modelos"][target]["uri"] == f"models:/prueba-demanda-{target}/2"


def test_archivo_exportado_predice_igual_que_el_registry(registry, monkeypatch):
    out = registry / "champion"
    export.export_champions(PARAMS, out)
    monkeypatch.setattr(loader, "MODELS_DIR", out)
    X = _features(50, seed=1)[ALL_FEATURES]

    exported = loader.load_pipelines()
    for target in ("salidas", "llegadas"):
        from_registry = predict.load_champion(PARAMS, target).predict(X)
        assert np.array_equal(exported[target].predict(X), from_registry)


def test_servicio_responde_con_el_modelo_exportado(registry, monkeypatch):
    out = registry / "champion"
    export.export_champions(PARAMS, out)
    monkeypatch.setattr(loader, "MODELS_DIR", out)
    from mibici.api.main import app

    payload = {
        "hora": 8, "dia_semana": 1, "mes": 4,
        "es_fin_de_semana": False, "es_hora_pico_manana": True, "es_hora_pico_tarde": False,
        "salidas_lag_24h": 3.0, "salidas_lag_48h": 2.5, "salidas_lag_168h": 4.0,
        "llegadas_lag_24h": 2.0, "llegadas_lag_168h": 3.5, "media_7d": 3.2,
        "salidas_sistema_lag_24h": 450.0, "capacidad": 15.0, "lat": 20.6736, "lon": -103.3435,
    }  # fmt: skip

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["salidas_pred"] > 0
    assert body["balance"] == pytest.approx(body["salidas_pred"] - body["llegadas_pred"])
