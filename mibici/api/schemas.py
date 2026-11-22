"""Modelos Pydantic de entrada y salida del servicio MiBici.

Implementacion completa. Los schemas definen el contrato de la API y son
usados por FastAPI para validacion automatica y generacion del Swagger.

Features del modelo (panel v3, train.py):
    NUMERIC_FEATURES (13): hora, dia_semana, mes, salidas_lag_24h,
        salidas_lag_48h, salidas_lag_168h, llegadas_lag_24h, llegadas_lag_168h,
        media_7d, salidas_sistema_lag_24h, capacidad, lat, lon.
    BOOL_FEATURES (3): es_fin_de_semana, es_hora_pico_manana, es_hora_pico_tarde.

Los lags son Optional[float] porque HistGBR los maneja como NaN de forma nativa
(estaciones con poca historia en el primer mes). El resto de features son
obligatorios.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Features de entrada para pronosticar la demanda de una estacion y hora.

    Todos los campos corresponden a columnas del panel v3
    (data/processed/demand_features_v3.parquet).
    """

    # --- Features de calendario (obligatorios) ---
    hora: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hora del dia (0-23).",
        examples=[8],
    )
    dia_semana: int = Field(
        ...,
        ge=0,
        le=6,
        description="Dia de la semana: 0=lunes, 6=domingo.",
        examples=[1],
    )
    mes: int = Field(
        ...,
        ge=1,
        le=12,
        description="Mes del anio (1-12).",
        examples=[4],
    )

    # --- Features booleanas de calendario (obligatorias) ---
    es_fin_de_semana: bool = Field(
        ...,
        description="True si el dia es sabado o domingo.",
        examples=[False],
    )
    es_hora_pico_manana: bool = Field(
        ...,
        description="True si la hora esta en el pico matutino (params.yaml horas_pico).",
        examples=[True],
    )
    es_hora_pico_tarde: bool = Field(
        ...,
        description="True si la hora esta en el pico vespertino (params.yaml horas_pico).",
        examples=[False],
    )

    # --- Features de rezago (Optional: None -> NaN, manejado por HistGBR) ---
    salidas_lag_24h: Optional[float] = Field(
        None,
        ge=0,
        description="Salidas en la misma hora del dia anterior. None si no hay historia.",
        examples=[3.0],
    )
    salidas_lag_48h: Optional[float] = Field(
        None,
        ge=0,
        description="Salidas en la misma hora de hace 48 h.",
        examples=[2.5],
    )
    salidas_lag_168h: Optional[float] = Field(
        None,
        ge=0,
        description="Salidas en la misma hora de la semana anterior (168 h).",
        examples=[4.0],
    )
    llegadas_lag_24h: Optional[float] = Field(
        None,
        ge=0,
        description="Llegadas en la misma hora del dia anterior.",
        examples=[2.0],
    )
    llegadas_lag_168h: Optional[float] = Field(
        None,
        ge=0,
        description="Llegadas en la misma hora de la semana anterior.",
        examples=[3.5],
    )
    media_7d: Optional[float] = Field(
        None,
        ge=0,
        description="Media de salidas en la misma hora de los 7 dias previos.",
        examples=[3.2],
    )
    salidas_sistema_lag_24h: Optional[float] = Field(
        None,
        ge=0,
        description="Total de salidas del sistema completo en la misma hora del dia anterior.",
        examples=[450.0],
    )

    # --- Features de estacion (obligatorias) ---
    capacidad: float = Field(
        ...,
        gt=0,
        description="Capacidad nominal de la estacion (numero de bicicletas).",
        examples=[15.0],
    )
    lat: float = Field(
        ...,
        ge=20.0,
        le=21.0,
        description="Latitud de la estacion (zona metropolitana de Guadalajara).",
        examples=[20.6736],
    )
    lon: float = Field(
        ...,
        ge=-104.0,
        le=-103.0,
        description="Longitud de la estacion.",
        examples=[-103.3435],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Pronostico de demanda para una estacion y hora.

    salidas_pred y llegadas_pred son siempre >= 0 (Poisson).
    balance = salidas_pred - llegadas_pred: positivo indica que la estacion
    se vaciara, negativo que se llenara.
    """

    salidas_pred: float = Field(
        ...,
        ge=0,
        description="Pronostico de salidas (bicicletas que salen) >= 0.",
        examples=[4.1],
    )
    llegadas_pred: float = Field(
        ...,
        ge=0,
        description="Pronostico de llegadas (bicicletas que llegan) >= 0.",
        examples=[2.8],
    )
    balance: float = Field(
        ...,
        description="Balance neto = salidas_pred - llegadas_pred. "
        "Positivo: la estacion tiende a vaciarse. "
        "Negativo: tiende a llenarse.",
        examples=[1.3],
    )


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: str = Field(..., examples=["ok"])
    source: str = Field(
        ...,
        description="Fuente de los modelos: 'local' (pkl en models/) o 'registry' (MLflow).",
        examples=["local"],
    )
    targets: list[str] = Field(
        ...,
        description="Objetivos pronosticados.",
        examples=[["salidas", "llegadas"]],
    )


class MetricsResponse(BaseModel):
    """Metricas de desempeno del modelo en validacion walk-forward.

    Fuente: reports/model_selection/walk_forward_6_cortes.csv
    (generado por make evaluate / dvc repro train).
    """

    mae_salidas: float = Field(
        ...,
        description="MAE promedio de salidas en los 6 cortes walk-forward.",
        examples=[0.674],
    )
    mae_llegadas: float = Field(
        ...,
        description="MAE promedio de llegadas en los 6 cortes walk-forward.",
        examples=[0.662],
    )
    mae_balance: float = Field(
        ...,
        description="MAE promedio del balance neto en los 6 cortes walk-forward.",
        examples=[0.716],
    )
    n_folds: int = Field(
        ...,
        description="Numero de cortes walk-forward evaluados.",
        examples=[6],
    )
    source: str = Field(
        ...,
        description="Archivo fuente de las metricas.",
        examples=["reports/model_selection/walk_forward_6_cortes.csv"],
    )
