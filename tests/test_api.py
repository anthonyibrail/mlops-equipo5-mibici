"""Tests del servicio FastAPI MiBici.

PENDIENTE (MLOps / Software Engineer):
    Implementar los tests marcados con @pytest.mark.skip una vez que
    mibici/api/loader.py y mibici/api/main.py esten completos.

Instrucciones generales:
    1. Importar TestClient de fastapi.testclient.
    2. Parchear mibici.api.loader.get_pipelines con un mock que retorne
       pipelines sklearn dummy (no necesita datos reales).
    3. Cada test tiene un comentario con la implementacion esperada.

Correr solo estos tests:
    pytest tests/test_api.py -v

Correr ignorando los skips para ver el estado:
    pytest tests/test_api.py -v --collect-only
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixture del cliente de prueba (PENDIENTE: descomentar cuando este listo)
# ---------------------------------------------------------------------------
#
# @pytest.fixture
# def client(monkeypatch):
#     """TestClient con modelo mockeado para no depender de los pkl reales."""
#     import numpy as np
#     from unittest.mock import MagicMock
#     from fastapi.testclient import TestClient
#     from mibici.api.main import app
#
#     mock_pipeline = MagicMock()
#     mock_pipeline.predict.return_value = np.array([3.5])
#
#     monkeypatch.setattr(
#         "mibici.api.loader.get_pipelines",
#         lambda: {"salidas": mock_pipeline, "llegadas": mock_pipeline},
#     )
#     return TestClient(app)
#
# PAYLOAD_VALIDO = {
#     "hora": 8, "dia_semana": 1, "mes": 4,
#     "es_fin_de_semana": False, "es_hora_pico_manana": True, "es_hora_pico_tarde": False,
#     "salidas_lag_24h": 3.0, "salidas_lag_48h": 2.5, "salidas_lag_168h": 4.0,
#     "llegadas_lag_24h": 2.0, "llegadas_lag_168h": 3.5,
#     "media_7d": 3.2, "salidas_sistema_lag_24h": 450.0,
#     "capacidad": 15.0, "lat": 20.6736, "lon": -103.3435,
# }

# ---------------------------------------------------------------------------
# Tests de GET /health
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="PENDIENTE: implementar GET /health en mibici/api/main.py")
def test_health_returns_200():
    """GET /health debe retornar 200 con status='ok'.

    TODO:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["source"] == "local"
        assert "salidas" in body["targets"]
        assert "llegadas" in body["targets"]
    """


@pytest.mark.skip(reason="PENDIENTE: implementar GET /health en mibici/api/main.py")
def test_health_response_schema():
    """GET /health debe retornar los campos definidos en HealthResponse.

    TODO:
        response = client.get("/health")
        body = response.json()
        assert set(body.keys()) == {"status", "source", "targets"}
    """


# ---------------------------------------------------------------------------
# Tests de POST /predict
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_valid_request_returns_200():
    """POST /predict con payload valido debe retornar 200.

    TODO:
        response = client.post("/predict", json=PAYLOAD_VALIDO)
        assert response.status_code == 200
    """


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_response_has_expected_fields():
    """POST /predict debe retornar salidas_pred, llegadas_pred y balance.

    TODO:
        response = client.post("/predict", json=PAYLOAD_VALIDO)
        body = response.json()
        assert "salidas_pred" in body
        assert "llegadas_pred" in body
        assert "balance" in body
    """


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_outputs_are_non_negative():
    """salidas_pred y llegadas_pred deben ser siempre >= 0 (perdida Poisson).

    TODO:
        response = client.post("/predict", json=PAYLOAD_VALIDO)
        body = response.json()
        assert body["salidas_pred"] >= 0
        assert body["llegadas_pred"] >= 0
    """


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_balance_equals_salidas_minus_llegadas():
    """balance debe ser exactamente salidas_pred - llegadas_pred.

    TODO:
        response = client.post("/predict", json=PAYLOAD_VALIDO)
        body = response.json()
        assert abs(body["balance"] - (body["salidas_pred"] - body["llegadas_pred"])) < 1e-6
    """


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_with_null_lags():
    """Los lags pueden ser null (None); no debe lanzar error.

    HistGBR maneja NaN de forma nativa en los lag features.

    TODO:
        payload_sin_lags = {**PAYLOAD_VALIDO,
            "salidas_lag_24h": None, "salidas_lag_48h": None,
            "salidas_lag_168h": None, "llegadas_lag_24h": None,
            "llegadas_lag_168h": None, "media_7d": None,
            "salidas_sistema_lag_24h": None,
        }
        response = client.post("/predict", json=payload_sin_lags)
        assert response.status_code == 200
    """


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_invalid_hora_returns_422():
    """hora=25 debe retornar 422 Unprocessable Entity (validacion Pydantic).

    TODO:
        payload_invalido = {**PAYLOAD_VALIDO, "hora": 25}
        response = client.post("/predict", json=payload_invalido)
        assert response.status_code == 422
    """


@pytest.mark.skip(reason="PENDIENTE: implementar POST /predict en mibici/api/main.py")
def test_predict_missing_required_field_returns_422():
    """Omitir 'hora' debe retornar 422.

    TODO:
        payload_incompleto = {k: v for k, v in PAYLOAD_VALIDO.items() if k != "hora"}
        response = client.post("/predict", json=payload_incompleto)
        assert response.status_code == 422
    """


# ---------------------------------------------------------------------------
# Tests de GET /metrics
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="PENDIENTE: implementar GET /metrics en mibici/api/main.py")
def test_metrics_returns_200():
    """GET /metrics debe retornar 200.

    TODO:
        response = client.get("/metrics")
        assert response.status_code == 200
    """


@pytest.mark.skip(reason="PENDIENTE: implementar GET /metrics en mibici/api/main.py")
def test_metrics_values_are_finite_and_positive():
    """Los MAE deben ser numeros positivos finitos.

    TODO:
        import math
        response = client.get("/metrics")
        body = response.json()
        for key in ("mae_salidas", "mae_llegadas", "mae_balance"):
            assert math.isfinite(body[key]), f"{key} no es finito"
            assert body[key] > 0, f"{key} debe ser positivo"
    """


@pytest.mark.skip(reason="PENDIENTE: implementar GET /metrics en mibici/api/main.py")
def test_metrics_n_folds_is_6():
    """n_folds debe ser 6 (walk-forward de 6 cortes).

    TODO:
        response = client.get("/metrics")
        body = response.json()
        assert body["n_folds"] == 6
    """


# ---------------------------------------------------------------------------
# Test que SI debe pasar ahora (valida que los schemas son correctos)
# ---------------------------------------------------------------------------


def test_schemas_importable():
    """Los schemas Pydantic deben ser importables sin errores.

    Este test pasa desde el primer commit del scaffold.
    """
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

    # Payload valido
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
    assert req.salidas_lag_24h is None  # lags opcionales

    # hora invalida
    with pytest.raises(ValidationError):
        PredictionRequest(
            hora=25,  # invalido: debe ser 0-23
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
