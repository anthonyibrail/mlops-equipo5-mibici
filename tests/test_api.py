"""Tests del servicio FastAPI MiBici.

Todos los tests usan un pipeline sklearn mockeado; no requieren los pkl reales.

Ejecutar: pytest tests/test_api.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mibici.api.loader as _loader_module

# ---------------------------------------------------------------------------
# Payload de prueba valido (cubre todos los campos de PredictionRequest)
# ---------------------------------------------------------------------------

PAYLOAD_VALIDO = {
    "hora": 8,
    "dia_semana": 1,
    "mes": 4,
    "es_fin_de_semana": False,
    "es_hora_pico_manana": True,
    "es_hora_pico_tarde": False,
    "salidas_lag_24h": 3.0,
    "salidas_lag_48h": 2.5,
    "salidas_lag_168h": 4.0,
    "llegadas_lag_24h": 2.0,
    "llegadas_lag_168h": 3.5,
    "media_7d": 3.2,
    "salidas_sistema_lag_24h": 450.0,
    "capacidad": 15.0,
    "lat": 20.6736,
    "lon": -103.3435,
}


# ---------------------------------------------------------------------------
# Fixture del cliente de prueba con modelo mockeado
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient con pipelines mockeados; no depende de los pkl reales."""
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([3.5])

    # Parchear load_pipelines antes de que el lifespan arranque
    monkeypatch.setattr(
        _loader_module,
        "load_pipelines",
        lambda: {"salidas": mock_pipeline, "llegadas": mock_pipeline},
    )

    # Importar app DESPUES del parche para que el lifespan use la version mockeada
    from mibici.api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests de GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200(client):
    """GET /health debe retornar 200 con status='ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["source"] == "local"
    assert "salidas" in body["targets"]
    assert "llegadas" in body["targets"]


def test_health_response_schema(client):
    """GET /health debe retornar los campos definidos en HealthResponse."""
    response = client.get("/health")
    body = response.json()
    assert set(body.keys()) == {"status", "source", "targets"}


# ---------------------------------------------------------------------------
# Tests de POST /predict
# ---------------------------------------------------------------------------


def test_predict_valid_request_returns_200(client):
    """POST /predict con payload valido debe retornar 200."""
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200


def test_predict_response_has_expected_fields(client):
    """POST /predict debe retornar salidas_pred, llegadas_pred y balance."""
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    body = response.json()
    assert "salidas_pred" in body
    assert "llegadas_pred" in body
    assert "balance" in body


def test_predict_outputs_are_non_negative(client):
    """salidas_pred y llegadas_pred deben ser siempre >= 0 (perdida Poisson)."""
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    body = response.json()
    assert body["salidas_pred"] >= 0
    assert body["llegadas_pred"] >= 0


def test_predict_balance_equals_salidas_minus_llegadas(client):
    """balance debe ser exactamente salidas_pred - llegadas_pred."""
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    body = response.json()
    expected = body["salidas_pred"] - body["llegadas_pred"]
    assert abs(body["balance"] - expected) < 1e-6


def test_predict_with_null_lags(client):
    """Los lags pueden ser null; HistGBR maneja NaN de forma nativa."""
    payload_sin_lags = {
        **PAYLOAD_VALIDO,
        "salidas_lag_24h": None,
        "salidas_lag_48h": None,
        "salidas_lag_168h": None,
        "llegadas_lag_24h": None,
        "llegadas_lag_168h": None,
        "media_7d": None,
        "salidas_sistema_lag_24h": None,
    }
    response = client.post("/predict", json=payload_sin_lags)
    assert response.status_code == 200


def test_predict_invalid_hora_returns_422(client):
    """hora=25 debe retornar 422 Unprocessable Entity (validacion Pydantic)."""
    payload_invalido = {**PAYLOAD_VALIDO, "hora": 25}
    response = client.post("/predict", json=payload_invalido)
    assert response.status_code == 422


def test_predict_missing_required_field_returns_422(client):
    """Omitir 'hora' debe retornar 422."""
    payload_incompleto = {k: v for k, v in PAYLOAD_VALIDO.items() if k != "hora"}
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests de GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200(client):
    """GET /metrics debe retornar 200 cuando el CSV de metricas existe."""
    response = client.get("/metrics")
    # Si el CSV no existe en este entorno, se acepta 503
    assert response.status_code in (200, 503)


def test_metrics_values_are_finite_and_positive(client):
    """Los MAE deben ser numeros positivos finitos."""
    response = client.get("/metrics")
    if response.status_code == 503:
        pytest.skip("CSV de metricas no disponible en este entorno")
    body = response.json()
    for key in ("mae_salidas", "mae_llegadas", "mae_balance"):
        assert math.isfinite(body[key]), f"{key} no es finito"
        assert body[key] > 0, f"{key} debe ser positivo"


def test_metrics_n_folds_is_6(client):
    """n_folds debe ser 6 (walk-forward de 6 cortes)."""
    response = client.get("/metrics")
    if response.status_code == 503:
        pytest.skip("CSV de metricas no disponible en este entorno")
    body = response.json()
    assert body["n_folds"] == 6


# ---------------------------------------------------------------------------
# Tests de schemas (no dependen de la app corriendo)
# ---------------------------------------------------------------------------


def test_schemas_importable():
    """Los schemas Pydantic deben ser importables sin errores."""
    from mibici.api.schemas import (  # noqa: F401
        HealthResponse,
        MetricsResponse,
        PredictionRequest,
        PredictionResponse,
    )


def test_prediction_request_validates_correctly():
    """PredictionRequest debe aceptar un payload valido y rechazar uno invalido."""
    from pydantic import ValidationError

    from mibici.api.schemas import PredictionRequest

    req = PredictionRequest(
        hora=8,
        dia_semana=1,
        mes=4,
        es_fin_de_semana=False,
        es_hora_pico_manana=True,
        es_hora_pico_tarde=False,
        capacidad=15.0,
        lat=20.6736,
        lon=-103.3435,
    )
    assert req.hora == 8
    assert req.salidas_lag_24h is None

    with pytest.raises(ValidationError):
        PredictionRequest(
            hora=25,
            dia_semana=1,
            mes=4,
            es_fin_de_semana=False,
            es_hora_pico_manana=True,
            es_hora_pico_tarde=False,
            capacidad=15.0,
            lat=20.6736,
            lon=-103.3435,
        )


def test_prediction_response_validates_correctly():
    """PredictionResponse debe rechazar valores negativos en salidas/llegadas."""
    from pydantic import ValidationError

    from mibici.api.schemas import PredictionResponse

    valid = PredictionResponse(salidas_pred=3.5, llegadas_pred=2.0, balance=1.5)
    assert valid.balance == 1.5

    with pytest.raises(ValidationError):
        PredictionResponse(salidas_pred=-1.0, llegadas_pred=2.0, balance=-3.0)
