"""Tests de ingenieria de features (datos sinteticos, sin disco ni MLflow).

Todas las pruebas usan DataFrames construidos en memoria para que sean rapidas
y no dependan de los archivos de datos del proyecto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    PEAK_HOURS_AFTERNOON,
    PEAK_HOURS_MORNING,
    build_features,
    hourly_demand,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_STATIONS = 3
N_HOURS = 300  # suficiente para cubrir lag de 168h + margen


@pytest.fixture
def small_stations() -> pd.DataFrame:
    """Catalogo minimo de 3 estaciones."""
    return pd.DataFrame({
        "estacion_id": [1, 2, 3],
        "nombre":      ["A", "B", "C"],
        "lat":         [20.670, 20.680, 20.690],
        "lon":         [-103.350, -103.360, -103.370],
        "capacidad":   [15, 20, 10],
    })


@pytest.fixture
def small_demand() -> pd.DataFrame:
    """Matriz 300h x 3 estaciones que empieza un lunes (2024-01-08).

    Los valores son enteros aleatorios [0, 9] con semilla fija para
    que los tests sean deterministas.
    """
    idx = pd.date_range("2024-01-08", periods=N_HOURS, freq="h", name="hora_inicio")
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 10, size=(N_HOURS, N_STATIONS))
    return pd.DataFrame(
        data, index=idx, columns=pd.Index([1, 2, 3], name="estacion_id")
    )


# ---------------------------------------------------------------------------
# Tests de hourly_demand
# ---------------------------------------------------------------------------

def test_hourly_demand_shape():
    """La matriz de salida tiene (n_horas_unicas, n_estaciones)."""
    trips = pd.DataFrame({
        "inicio":     pd.date_range("2024-01-01", periods=24, freq="h"),
        "origen_id":  [1] * 12 + [2] * 12,
    })
    demand = hourly_demand(trips)
    assert demand.shape == (24, 2)


def test_hourly_demand_zero_fills_gaps():
    """Las horas sin viajes se rellenan con 0 en la salida."""
    trips = pd.DataFrame({
        "inicio":    [pd.Timestamp("2024-01-01 06:00"),
                      pd.Timestamp("2024-01-01 09:00")],
        "origen_id": [1, 1],
    })
    demand = hourly_demand(trips)
    # Hay 4 horas entre 06:00 y 09:00 inclusive
    assert demand.shape[0] == 4
    assert demand.loc[pd.Timestamp("2024-01-01 07:00"), 1] == 0
    assert demand.loc[pd.Timestamp("2024-01-01 08:00"), 1] == 0


def test_hourly_demand_index_is_hourly():
    """El indice tiene frecuencia horaria continua (sin saltos)."""
    trips = pd.DataFrame({
        "inicio":    [pd.Timestamp("2024-01-01 00:00"),
                      pd.Timestamp("2024-01-01 05:00")],
        "origen_id": [1, 1],
    })
    demand = hourly_demand(trips)
    diffs = demand.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_hourly_demand_index_name():
    """El indice se llama 'hora_inicio' y las columnas 'estacion_id'."""
    trips = pd.DataFrame({
        "inicio":    pd.date_range("2024-01-01", periods=3, freq="h"),
        "origen_id": [1, 2, 1],
    })
    demand = hourly_demand(trips)
    assert demand.index.name == "hora_inicio"
    assert demand.columns.name == "estacion_id"


# ---------------------------------------------------------------------------
# Tests de build_features: estructura de salida
# ---------------------------------------------------------------------------

def test_output_columns_present(small_demand, small_stations):
    """Todas las columnas esperadas existen en el DataFrame de salida."""
    result = build_features(small_demand, small_stations)
    expected = [
        "hora_inicio", "estacion_id",
        "demanda", "demanda_lag_1h", "demanda_lag_24h",
        "demanda_lag_168h", "media_7d", "demanda_sistema_lag_1h",
        "hora", "dia_semana", "mes",
        "es_fin_de_semana", "es_hora_pico_manana", "es_hora_pico_tarde",
        "capacidad", "lat", "lon",
    ]
    missing = [c for c in expected if c not in result.columns]
    assert not missing, f"Columnas faltantes: {missing}"


def test_row_count(small_demand, small_stations):
    """El total de filas es n_horas * n_estaciones."""
    result = build_features(small_demand, small_stations)
    assert len(result) == N_HOURS * N_STATIONS


def test_station_features_populated(small_demand, small_stations):
    """Los atributos de estacion no tienen nulos."""
    result = build_features(small_demand, small_stations)
    for col in ("capacidad", "lat", "lon"):
        assert result[col].notna().all(), f"Nulos inesperados en {col}"


def test_station_feature_values(small_demand, small_stations):
    """Los valores de capacidad de la estacion 2 son correctos."""
    result = build_features(small_demand, small_stations)
    st2 = result[result["estacion_id"] == 2]
    assert (st2["capacidad"] == 20).all()


# ---------------------------------------------------------------------------
# Tests de no-filtracion (data leakage)
# ---------------------------------------------------------------------------

def _station_series(result: pd.DataFrame, sid: int) -> pd.DataFrame:
    """Devuelve las filas de una estacion ordenadas por tiempo."""
    return (result[result["estacion_id"] == sid]
            .sort_values("hora_inicio")
            .reset_index(drop=True))


def test_no_leakage_lag_1h(small_demand, small_stations):
    """demanda_lag_1h en t debe ser igual a demanda en t-1."""
    r = _station_series(build_features(small_demand, small_stations), sid=1)
    assert r.loc[5, "demanda_lag_1h"] == pytest.approx(r.loc[4, "demanda"])
    assert r.loc[20, "demanda_lag_1h"] == pytest.approx(r.loc[19, "demanda"])


def test_no_leakage_lag_24h(small_demand, small_stations):
    """demanda_lag_24h en t debe ser igual a demanda en t-24."""
    r = _station_series(build_features(small_demand, small_stations), sid=1)
    assert r.loc[30, "demanda_lag_24h"] == pytest.approx(r.loc[6, "demanda"])


def test_no_leakage_lag_168h(small_demand, small_stations):
    """demanda_lag_168h en t debe ser igual a demanda en t-168."""
    r = _station_series(build_features(small_demand, small_stations), sid=1)
    # t=170 -> lag_168h = demanda[2]
    assert r.loc[170, "demanda_lag_168h"] == pytest.approx(r.loc[2, "demanda"])


def test_lag_168h_nan_for_first_168_rows(small_demand, small_stations):
    """Las primeras 168 filas no tienen historia suficiente -> NaN en lag_168h."""
    r = _station_series(build_features(small_demand, small_stations), sid=1)
    assert r["demanda_lag_168h"].iloc[:168].isna().all()


def test_media_7d_uses_only_past(small_demand, small_stations):
    """media_7d en t no incluye demanda[t]; media_7d[1] = demanda[0]."""
    r = _station_series(build_features(small_demand, small_stations), sid=1)
    # media_7d usa shift(1) antes del rolling, por lo que en t=1 solo ve demand[0]
    assert r.loc[1, "media_7d"] == pytest.approx(r.loc[0, "demanda"])


# ---------------------------------------------------------------------------
# Tests de features temporales
# ---------------------------------------------------------------------------

def test_temporal_weekend_flag(small_demand, small_stations):
    """es_fin_de_semana=True para sabado/domingo, False para lunes."""
    result = build_features(small_demand, small_stations)
    # 2024-01-08 es lunes
    mon = result[result["hora_inicio"] == pd.Timestamp("2024-01-08 00:00")]
    assert not mon["es_fin_de_semana"].any()
    # 2024-01-13 es sabado (5 dias despues del lunes)
    sat = result[result["hora_inicio"] == pd.Timestamp("2024-01-13 00:00")]
    assert sat["es_fin_de_semana"].all()


def test_temporal_morning_peak(small_demand, small_stations):
    """es_hora_pico_manana=True solo para horas en PEAK_HOURS_MORNING."""
    result = build_features(small_demand, small_stations)
    for h in PEAK_HOURS_MORNING:
        assert result.loc[result["hora"] == h, "es_hora_pico_manana"].all()
    off_peak = result[~result["hora"].isin(PEAK_HOURS_MORNING)]
    assert not off_peak["es_hora_pico_manana"].any()


def test_temporal_afternoon_peak(small_demand, small_stations):
    """es_hora_pico_tarde=True solo para horas en PEAK_HOURS_AFTERNOON."""
    result = build_features(small_demand, small_stations)
    for h in PEAK_HOURS_AFTERNOON:
        assert result.loc[result["hora"] == h, "es_hora_pico_tarde"].all()
    off_peak = result[~result["hora"].isin(PEAK_HOURS_AFTERNOON)]
    assert not off_peak["es_hora_pico_tarde"].any()


def test_temporal_hour_range(small_demand, small_stations):
    """La columna 'hora' contiene solo valores entre 0 y 23."""
    result = build_features(small_demand, small_stations)
    assert result["hora"].between(0, 23).all()


def test_temporal_day_of_week_range(small_demand, small_stations):
    """La columna 'dia_semana' contiene solo valores entre 0 y 6."""
    result = build_features(small_demand, small_stations)
    assert result["dia_semana"].between(0, 6).all()


# ---------------------------------------------------------------------------
# Test de demanda del sistema
# ---------------------------------------------------------------------------

def test_sistema_lag_1h_is_row_sum(small_demand, small_stations):
    """demanda_sistema_lag_1h en t = suma de demanda de todas las estaciones en t-1."""
    result = build_features(small_demand, small_stations)
    t4 = small_demand.index[4]
    t5 = small_demand.index[5]
    expected = float(small_demand.loc[t4].sum())
    row = result[(result["hora_inicio"] == t5) & (result["estacion_id"] == 1)]
    assert row["demanda_sistema_lag_1h"].iloc[0] == pytest.approx(expected)
