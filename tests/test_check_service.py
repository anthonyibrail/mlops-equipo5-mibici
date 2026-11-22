"""Pruebas de la verificacion de reproducibilidad del servicio (check_service.py).

Usan un Pipeline real de train.py entrenado con filas sinteticas, servido con
FastAPI en memoria (TestClient): comprueban el filtro del esquema, la conversion a
JSON y que las predicciones del servicio coinciden con las locales.
"""

import math
import pickle

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici.api import loader
from mibici.modeling import check_service as cs
from mibici.modeling.train import ALL_FEATURES, BOOL_FEATURES, build_pipeline

TARGETS = ["salidas", "llegadas"]


def _panel(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) * 5 for c in ALL_FEATURES})
    for c in BOOL_FEATURES:
        df[c] = rng.random(n) > 0.5
    df["hora"] = rng.integers(0, 24, n)
    df["dia_semana"] = rng.integers(0, 7, n)
    df["mes"] = rng.integers(1, 13, n)
    df["capacidad"] = rng.integers(5, 30, n).astype(float)
    df["lat"] = 20.6 + rng.random(n) * 0.1
    df["lon"] = -103.4 + rng.random(n) * 0.1
    df["salidas"] = rng.poisson(2, n)
    df["llegadas"] = rng.poisson(2, n)
    return df


def test_filtro_del_esquema_de_la_api():
    df = _panel(4)
    df.loc[0, "capacidad"] = 0  # estacion retirada
    df.loc[1, "lat"] = float("nan")
    df.loc[2, "lon"] = -99.0  # fuera de Guadalajara
    assert cs.valid_for_api(df).tolist() == [False, False, False, True]


def test_payload_convierte_tipos_y_nulos():
    row = _panel(1).iloc[0].copy()
    row["media_7d"] = float("nan")
    payload = cs.to_payload(row)
    assert payload["media_7d"] is None
    assert isinstance(payload["hora"], int)
    assert all(isinstance(payload[c], bool) for c in BOOL_FEATURES)
    assert set(payload) == set(ALL_FEATURES)


def test_compare_detecta_diferencias():
    idx = range(3)
    actual = pd.DataFrame({t: [1.0, 2.0, 3.0] for t in TARGETS}, index=idx)
    local = pd.DataFrame({t: [1.5, 2.0, 2.5] for t in TARGETS}, index=idx)
    same = cs.compare(local, local.copy(), actual, TARGETS)
    assert same["reproducible"] and same["salidas_filas_identicas"] == 3
    assert same["salidas_mae_local"] == pytest.approx(same["salidas_mae_servicio"])

    other = local.copy()
    other.loc[0, "salidas"] += 0.1
    diff = cs.compare(local, other, actual, TARGETS)
    assert not diff["reproducible"]
    assert diff["salidas_dif_max"] == pytest.approx(0.1)


def test_servicio_reproduce_las_predicciones_locales(tmp_path, monkeypatch):
    train = _panel(300)
    for target, scale in (("salidas", 1.0), ("llegadas", 2.0)):
        pipe = build_pipeline({"loss": "poisson", "max_iter": 20, "random_state": 42})
        pipe.fit(train[ALL_FEATURES], scale * train["hora"] + 1)
        with open(tmp_path / f"pipeline_{target}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)
    monkeypatch.setattr(loader, "MODELS_DIR", tmp_path)
    models = loader.load_pipelines()
    from mibici.api.main import app

    sample = _panel(40, seed=1)
    sample.loc[sample.index[0], "media_7d"] = float("nan")  # un nulo viaja como null
    local = pd.DataFrame(
        {t: np.clip(models[t].predict(sample[ALL_FEATURES]), 0, None) for t in TARGETS},
        index=sample.index,
    )
    with TestClient(app) as client:

        def post(payload):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            return response.json()

        service = cs.service_predictions(sample, post, TARGETS)

    summary = cs.compare(local, service, sample[TARGETS], TARGETS)
    assert summary["reproducible"]
    assert summary["salidas_filas_identicas"] == len(sample)
    assert not math.isnan(summary["llegadas_mae_servicio"])
