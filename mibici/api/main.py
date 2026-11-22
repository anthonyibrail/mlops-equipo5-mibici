"""Servicio REST de pronostico de demanda MiBici (FastAPI).

PENDIENTE (Feature / ML Engineer):
    Completar los tres endpoints reemplazando los HTTP 501 con la logica real.
    Ver los comentarios TODO en cada endpoint.

Para arrancar el servicio:
    uvicorn mibici.api.main:app --reload --port 8000

Para ver el Swagger interactivo:
    http://localhost:8000/docs

Endpoints:
    GET  /health   -- estado del servicio y modelos cargados
    POST /predict  -- pronostico de salidas, llegadas y balance
    GET  /metrics  -- MAE walk-forward del modelo @champion
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from mibici.api.schemas import HealthResponse, MetricsResponse, PredictionRequest, PredictionResponse

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
)

# ---------------------------------------------------------------------------
# TODO: cargar modelos al arrancar (lifespan o startup event)
#
# Opcion A (FastAPI >= 0.95, recomendada):
#   from contextlib import asynccontextmanager
#   @asynccontextmanager
#   async def lifespan(app: FastAPI):
#       from mibici.api.loader import load_pipelines
#       app.state.pipelines = load_pipelines()
#       yield
#   app = FastAPI(..., lifespan=lifespan)
#
# Opcion B (startup event, compatible con versiones anteriores):
#   @app.on_event("startup")
#   def startup():
#       from mibici.api.loader import load_pipelines
#       app.state.pipelines = load_pipelines()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Verifica que el servicio esta activo y los modelos estan cargados.",
)
def health() -> HealthResponse:
    """Retorna el estado del servicio.

    TODO:
        1. Comprobar que app.state.pipelines tiene las claves "salidas" y "llegadas".
        2. Retornar HealthResponse con status="ok", source="local",
           targets=list(app.state.pipelines.keys()).

    Ejemplo de respuesta esperada:
        {"status": "ok", "source": "local", "targets": ["salidas", "llegadas"]}
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "GET /health no implementado. "
            "Ver mibici/api/main.py y mibici/api/loader.py."
        ),
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
def predict(request: PredictionRequest) -> PredictionResponse:
    """Pronostica la demanda de bicicletas para una estacion y hora.

    TODO:
        1. Acceder a los pipelines: pipelines = app.state.pipelines
        2. Construir un DataFrame con las columnas de ALL_FEATURES
           (mibici/modeling/train.py) en el orden correcto.
           Los booleans deben ser bool (el ColumnTransformer aplica bool_to_int).
        3. Llamar a cada pipeline:
               import numpy as np
               salidas_pred = float(np.clip(pipelines["salidas"].predict(X), 0, None)[0])
               llegadas_pred = float(np.clip(pipelines["llegadas"].predict(X), 0, None)[0])
        4. Calcular balance = salidas_pred - llegadas_pred.
        5. Retornar PredictionResponse(
               salidas_pred=salidas_pred,
               llegadas_pred=llegadas_pred,
               balance=balance,
           )

    Referencia:
        - mibici/modeling/train.py    -> ALL_FEATURES, NUMERIC_FEATURES, BOOL_FEATURES
        - mibici/modeling/predict.py  -> logica de inferencia existente
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "POST /predict no implementado. "
            "Ver mibici/api/main.py para las instrucciones."
        ),
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
    """Retorna las metricas de desempeno del modelo en validacion walk-forward.

    TODO:
        1. Leer reports/model_selection/walk_forward_6_cortes.csv con pandas.
        2. Calcular el MAE promedio de los 6 cortes para cada objetivo.
        3. Calcular el MAE de balance = MAE de (salidas_pred - llegadas_pred)
           agrupado por corte y luego promediado.
        4. Retornar MetricsResponse con los valores calculados.

    Esquema del CSV (walk_forward_6_cortes.csv):
        objetivo, corte, mae, rmse, mae_pico, tipo_corte

    Ejemplo de implementacion:
        import pandas as pd
        from pathlib import Path
        REPORT = Path("reports/model_selection/walk_forward_6_cortes.csv")
        if not REPORT.exists():
            raise HTTPException(status_code=503, detail="Metricas no disponibles.")
        df = pd.read_csv(REPORT)
        mae_salidas = float(df[df["objetivo"] == "salidas"]["mae"].mean())
        mae_llegadas = float(df[df["objetivo"] == "llegadas"]["mae"].mean())
        # balance: calcular por corte y promediar
        # (ver mibici/modeling/validation.py -> net_balance_metrics)
        ...

    Referencia:
        - reports/model_selection/walk_forward_6_cortes.csv
        - mibici/modeling/validation.py -> net_balance_metrics()
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "GET /metrics no implementado. "
            "Ver mibici/api/main.py para las instrucciones."
        ),
    )
