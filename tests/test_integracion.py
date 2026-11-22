"""Prueba de integracion de punta a punta del pipeline MiBici.

Recorre el mismo camino que los datos reales, con archivos sinteticos pequenos:

    CSV crudos del portal -> carga (v0) -> limpieza (v1) -> panel v3 de features
    -> entrenamiento del Pipeline -> servicio FastAPI (POST /predict) -> metricas

Comprueba que cada etapa entrega lo que la siguiente necesita y que el servicio
devuelve exactamente las predicciones del modelo entrenado. No requiere descargar
datos ni tener MLflow corriendo.
"""

import pickle

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici.api import loader
from mibici.cleaning import TripCleaner
from mibici.dataset import STATIONS_FILE, MiBiciLoader
from mibici.features import build_features_v3, hourly_arrivals, hourly_demand
from mibici.modeling.check_service import service_predictions, valid_for_api
from mibici.modeling.train import ALL_FEATURES, build_pipeline
from mibici.modeling.validation import regression_metrics

TARGETS = ["salidas", "llegadas"]
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
STATIONS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "name": ["(GDL-001) A", "(GDL-002) B", "(GDL-003) C", "(GDL-004) D"],
        "latitude": [20.670, 20.675, 20.680, 20.690],
        "longitude": [-103.350, -103.345, -103.340, -103.330],
        "dpcapacity": [15, 20, 12, 18],
    }
)


def _write_raw_month(path, start: str, days: int, rng, first_id: int) -> int:
    """Escribe un CSV mensual con el formato del portal: viajes validos y algunos sucios."""
    rows, trip_id = [], first_id
    for hour in pd.date_range(start, periods=days * 24, freq="h"):
        # Mas viajes en las horas pico, para que el modelo tenga un patron que aprender
        n = rng.poisson(4 if hour.hour in PEAK_HOURS else 1)
        for _ in range(n):
            origin, dest = rng.choice([1, 2, 3, 4], size=2, replace=False)
            start_t = hour + pd.Timedelta(minutes=int(rng.integers(0, 50)))
            end_t = start_t + pd.Timedelta(minutes=int(rng.integers(10, 25)))
            rows.append([trip_id, 100 + trip_id % 50, "M", 1990, start_t, end_t, origin, dest])
            trip_id += 1
    t0 = pd.Timestamp(start) + pd.Timedelta(hours=10)
    rows += [
        rows[0],  # duplicado
        [trip_id, 1, "F", 1985, t0, t0 + pd.Timedelta(minutes=1), 1, 1],  # viaje fantasma
        [trip_id + 1, 1, "", 1985, t0, t0 + pd.Timedelta(minutes=10), 1, 99],  # estacion 99
    ]
    df = pd.DataFrame(
        rows,
        columns=["Viaje_Id", "Usuario_Id", "Genero", "Anio", "Inicio", "Fin", "Origen", "Destino"],
    )
    for col in ("Inicio", "Fin"):
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(path, index=False)
    return trip_id + 2


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(42)
    raw = tmp_path / "raw"
    raw.mkdir()
    STATIONS.to_csv(raw / STATIONS_FILE, index=False)
    next_id = _write_raw_month(raw / "datos_abiertos_2026_02.csv", "2026-02-01", 28, rng, 1)
    _write_raw_month(raw / "datos_abiertos_2026_03.csv", "2026-03-01", 21, rng, next_id)
    return raw


def test_pipeline_de_punta_a_punta(raw_dir, tmp_path, monkeypatch):
    # 1) Carga: CSV crudos -> v0
    data = MiBiciLoader(raw_dir)
    v0 = data.load_trips()
    stations = data.load_stations()
    assert set(v0["archivo"].unique()) == {"2026_02", "2026_03"}

    # 2) Limpieza: v0 -> v1 (elimina duplicados, viajes fantasma y estaciones desconocidas)
    cleaner = TripCleaner(stations)
    v1 = cleaner.clean(v0)
    removed = {r["regla"]: r["filas_afectadas"] for r in cleaner.report}
    assert removed["duplicados"] == 2
    assert removed["viaje_fantasma"] == 2
    assert removed["estacion_desconocida"] == 2
    eliminated = sum(r["filas_afectadas"] for r in cleaner.report if r["accion"] == "eliminar")
    assert eliminated == 6  # solo los registros sucios que se insertaron
    assert len(v1) == len(v0) - eliminated

    # 3) Features: v1 -> panel v3 con salidas, llegadas y rezagos de al menos 24 h
    panel = build_features_v3(hourly_demand(v1), hourly_arrivals(v1), stations)
    assert set(ALL_FEATURES + TARGETS) <= set(panel.columns)
    assert "salidas_lag_1h" not in panel.columns
    assert panel["salidas"].sum() == len(v1)

    # 4) Entrenamiento: febrero y marzo hasta el dia 14 / prueba: el resto de marzo
    cutoff = pd.Timestamp("2026-03-15")
    train = panel[panel["hora_inicio"] < cutoff].dropna(subset=["salidas_lag_168h"])
    test = panel[(panel["hora_inicio"] >= cutoff) & valid_for_api(panel)]
    models = {}
    for target in TARGETS:
        pipe = build_pipeline({"loss": "poisson", "max_iter": 50, "random_state": 42})
        models[target] = pipe.fit(train[ALL_FEATURES], train[target])
        with open(tmp_path / f"pipeline_{target}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)
    local = pd.DataFrame(
        {t: np.clip(models[t].predict(test[ALL_FEATURES]), 0, None) for t in TARGETS},
        index=test.index,
    )

    # 5) Servicio: el modelo guardado se sirve con FastAPI y se consulta por POST /predict
    monkeypatch.setattr(loader, "MODELS_DIR", tmp_path)
    from mibici.api.main import app

    with TestClient(app) as client:

        def post(payload):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            return response.json()

        served = service_predictions(test, post, TARGETS)
    pd.testing.assert_frame_equal(served, local, check_exact=True)

    # 6) Metricas: el modelo servido supera a un pronostico constante (media del entrenamiento)
    for target in TARGETS:
        metrics = regression_metrics(test[target], served[target], test["hora"], PEAK_HOURS)
        constant = np.full(len(test), train[target].mean())
        reference = regression_metrics(test[target], constant, test["hora"], PEAK_HOURS)
        assert np.isfinite(metrics["mae"]) and np.isfinite(metrics["mae_pico"])
        assert metrics["mae"] < reference["mae"]
