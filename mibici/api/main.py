"""Servicio REST de pronostico de demanda MiBici (FastAPI).

Endpoints:
    GET  /health   -- estado del servicio y modelos cargados
    POST /predict  -- pronostico de salidas, llegadas y balance
    GET  /metrics  -- MAE walk-forward del modelo en produccion

Para arrancar el servicio:
    uvicorn mibici.api.main:app --reload --port 8000

Para ver el Swagger interactivo:
    http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from mibici.api import loader as _loader
from mibici.api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from mibici.config import REPORTS_DIR
from mibici.modeling.train import ALL_FEATURES

REPORT_PATH = REPORTS_DIR / "model_selection" / "walk_forward_6_cortes.csv"

# ---------------------------------------------------------------------------
# Lifespan: carga los modelos al arrancar la app y los almacena en app.state
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Carga los pipelines pkl al iniciar el servicio."""
    application.state.pipelines = _loader.load_pipelines()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MiBici Demand Forecast API",
    description=(
        "Pronostica salidas y llegadas de bicicletas por estacion y hora "
        "usando el modelo HistGradientBoosting entrenado con el pipeline DVC. "
        "El modelo se carga desde models/ al arrancar el servicio."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Verifica que el servicio esta activo y los modelos estan cargados.",
)
def health(request: Request) -> HealthResponse:
    """Retorna el estado del servicio y los objetivos disponibles."""
    pipelines = request.app.state.pipelines
    return HealthResponse(
        status="ok",
        source="local",
        targets=list(pipelines.keys()),
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Pronostico de demanda",
    description=(
        "Recibe los 16 features del panel v3 y retorna salidas_pred, "
        "llegadas_pred y balance (salidas - llegadas). "
        "Los lags pueden ser null (None); HistGBR los maneja como NaN."
    ),
)
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """Pronostica la demanda de bicicletas para una estacion y hora."""
    pipelines = request.app.state.pipelines

    row = {
        "hora": body.hora,
        "dia_semana": body.dia_semana,
        "mes": body.mes,
        "salidas_lag_24h": body.salidas_lag_24h,
        "salidas_lag_48h": body.salidas_lag_48h,
        "salidas_lag_168h": body.salidas_lag_168h,
        "llegadas_lag_24h": body.llegadas_lag_24h,
        "llegadas_lag_168h": body.llegadas_lag_168h,
        "media_7d": body.media_7d,
        "salidas_sistema_lag_24h": body.salidas_sistema_lag_24h,
        "capacidad": body.capacidad,
        "lat": body.lat,
        "lon": body.lon,
        "es_fin_de_semana": body.es_fin_de_semana,
        "es_hora_pico_manana": body.es_hora_pico_manana,
        "es_hora_pico_tarde": body.es_hora_pico_tarde,
    }
    X = pd.DataFrame([row])[ALL_FEATURES]

    salidas_pred = float(np.clip(pipelines["salidas"].predict(X), 0.0, None)[0])
    llegadas_pred = float(np.clip(pipelines["llegadas"].predict(X), 0.0, None)[0])
    balance = salidas_pred - llegadas_pred

    return PredictionResponse(
        salidas_pred=salidas_pred,
        llegadas_pred=llegadas_pred,
        balance=balance,
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Metricas del modelo",
    description=(
        "Retorna el MAE walk-forward promedio (6 cortes) para salidas, "
        "llegadas y balance. Fuente: reports/model_selection/walk_forward_6_cortes.csv."
    ),
)
def metrics() -> MetricsResponse:
    """Retorna las metricas de desempeno del modelo en validacion walk-forward."""
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metricas no disponibles. Ejecuta 'dvc repro train' o 'make evaluate' primero.",
        )

    df = pd.read_csv(REPORT_PATH)

    mae_salidas = float(df[df["objetivo"] == "salidas"]["mae"].mean())
    mae_llegadas = float(df[df["objetivo"] == "llegadas"]["mae"].mean())
    mae_balance = float(df[df["objetivo"] == "balance"]["mae"].mean())
    n_folds = int(df[df["objetivo"] == "salidas"]["corte"].nunique())

    return MetricsResponse(
        mae_salidas=mae_salidas,
        mae_llegadas=mae_llegadas,
        mae_balance=mae_balance,
        n_folds=n_folds,
        source=str(REPORT_PATH),
    )
