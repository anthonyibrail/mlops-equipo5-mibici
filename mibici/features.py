"""Feature engineering: viajes limpios -> panel de demanda horaria con features.

Produce dos versiones del panel:
  v2 (legacy): columna 'demanda' (salidas), lags de 1/24/168h, media_7d con
               shift(1), sistema_lag_1h.  Se mantiene para compatibilidad.
  v3 (actual): columnas 'salidas' y 'llegadas' separadas; todos los lags
               >= 24h para respetar el horizonte de pronostico (a las 00:00
               del dia D se predicen las 24h de D usando datos de D-1 o antes).

Regla de no-filtracion (data leakage):
  - v2: todos los lags usan shift >= 1 (no incluye el instante t).
  - v3: todos los lags usan shift >= 24 para el horizonte de 24h.

Uso:
    python -m mibici.features main       # genera panel v2 (legacy)
    python -m mibici.features build-v3   # genera panel v3 (Fase 2)
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, load_params
from mibici.dataset import MiBiciLoader

app = typer.Typer()

PEAK_HOURS_MORNING = {7, 8, 9}
PEAK_HOURS_AFTERNOON = {17, 18, 19}
ROLLING_WINDOW_H = 7 * 24  # 168 horas = 7 dias


# ---------------------------------------------------------------------------
# Funciones base de agregacion
# ---------------------------------------------------------------------------


def hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz (hora x estacion) con conteo de viajes que INICIAN.

    El indice es un DatetimeIndex completo (sin huecos) a frecuencia horaria.
    Las horas sin viajes se rellenan con 0.

    Args:
        df: DataFrame con columnas 'inicio' (datetime) y 'origen_id' (int).

    Returns:
        DataFrame con index.name='hora_inicio' y columns.name='estacion_id'.
    """
    hours = df["inicio"].dt.floor("h")
    counts = df.groupby([hours, "origen_id"]).size().unstack(fill_value=0)
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="h", name="hora_inicio")
    result = counts.reindex(full, fill_value=0)
    result.columns.name = "estacion_id"
    return result


def hourly_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz (hora x estacion) con conteo de viajes que TERMINAN en cada estacion.

    Reutiliza hourly_demand renombrando 'fin' -> 'inicio' y 'destino_id' ->
    'origen_id' para contar llegadas con la misma logica que las salidas.
    Las horas sin llegadas se rellenan con 0.

    Args:
        df: DataFrame con columnas 'fin' (datetime) y 'destino_id' (int).
            Se descartan filas con NaN en esas columnas antes de agregar.

    Returns:
        DataFrame con index.name='hora_inicio' y columns.name='estacion_id'.
    """
    valid = df[["fin", "destino_id"]].dropna()
    renamed = valid.rename(columns={"fin": "inicio", "destino_id": "origen_id"})
    return hourly_demand(renamed)


# ---------------------------------------------------------------------------
# Panel v2 (legacy) - se mantiene sin cambios para compatibilidad con tests
# ---------------------------------------------------------------------------


def build_features(demand: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye el panel v2 a partir de la matriz de demanda horaria.

    Columna objetivo: 'demanda'. Incluye rezagos de 1h, 24h y 168h.
    Mantenida para compatibilidad con la Fase 1 y las pruebas existentes.

    Args:
        demand:   Matriz (DatetimeIndex x estacion_id) producida por hourly_demand().
                  index.name debe ser 'hora_inicio' y columns.name 'estacion_id'.
        stations: DataFrame con columnas estacion_id, capacidad, lat, lon.

    Returns:
        DataFrame largo, una fila por (hora_inicio, estacion_id), con columnas v2.
    """
    # --- 1. Features de rezago en formato ancho ---
    wide = {
        "demanda": demand,
        "demanda_lag_1h": demand.shift(1),
        "demanda_lag_24h": demand.shift(24),
        "demanda_lag_168h": demand.shift(168),
        "media_7d": demand.shift(1).rolling(window=ROLLING_WINDOW_H, min_periods=1).mean(),
    }
    system_lag_1h = demand.sum(axis=1).shift(1)

    # --- 2. Stack a formato largo ---
    parts = [wide_df.stack().rename(col) for col, wide_df in wide.items()]
    df = pd.concat(parts, axis=1).reset_index()
    df["demanda_sistema_lag_1h"] = df["hora_inicio"].map(system_lag_1h)

    # --- 3. Features temporales ---
    df["hora"] = df["hora_inicio"].dt.hour.astype("int8")
    df["dia_semana"] = df["hora_inicio"].dt.day_of_week.astype("int8")
    df["mes"] = df["hora_inicio"].dt.month.astype("int8")
    df["es_fin_de_semana"] = df["dia_semana"] >= 5
    df["es_hora_pico_manana"] = df["hora"].isin(PEAK_HOURS_MORNING)
    df["es_hora_pico_tarde"] = df["hora"].isin(PEAK_HOURS_AFTERNOON)

    # --- 4. Atributos estaticos de la estacion ---
    st_cols = stations.set_index("estacion_id")[["capacidad", "lat", "lon"]]
    df = df.merge(st_cols, left_on="estacion_id", right_index=True, how="left")

    df = df.sort_values(["estacion_id", "hora_inicio"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Panel v3 (Fase 2) - salidas y llegadas, lags >= 24h
# ---------------------------------------------------------------------------


def build_features_v3(
    salidas: pd.DataFrame,
    llegadas: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el panel v3 con salidas y llegadas separadas, lags >= 24h.

    Cumple la regla de no-filtracion del horizonte de pronostico a 24h:
    ningun feature incluye informacion de menos de 24 horas antes del
    instante t que se quiere pronosticar.

    Columnas objetivo: 'salidas' y 'llegadas'.
    Features de rezago: {salidas,llegadas}_lag_{24,48,168}h.
    Media movil: media_7d (168h de salidas con shift de 24h).
    Sistema:     salidas_sistema_lag_24h (total de salidas en t-24).
    Temporales:  hora, dia_semana, mes, flags de pico y fin de semana.
    Estaticos:   capacidad, lat, lon.

    Args:
        salidas:  Matriz wide (DatetimeIndex x estacion_id) de salidas.
                  Producida por hourly_demand().
        llegadas: Matriz wide (DatetimeIndex x estacion_id) de llegadas.
                  Producida por hourly_arrivals(). Debe cubrir el mismo
                  rango de tiempo que salidas (se alinean automaticamente).
        stations: DataFrame con columnas estacion_id, capacidad, lat, lon.

    Returns:
        DataFrame largo, una fila por (hora_inicio, estacion_id).
    """
    # Alinear rangos temporales y columnas de estacion
    idx = salidas.index.union(llegadas.index)
    cols = salidas.columns.union(llegadas.columns)
    salidas = salidas.reindex(index=idx, columns=cols, fill_value=0)
    llegadas = llegadas.reindex(index=idx, columns=cols, fill_value=0)

    # --- 1. Features de rezago en formato ancho (shift >= 24h) ---
    wide = {
        "salidas": salidas,
        "llegadas": llegadas,
        "salidas_lag_24h": salidas.shift(24),
        "salidas_lag_48h": salidas.shift(48),
        "salidas_lag_168h": salidas.shift(168),
        "llegadas_lag_24h": llegadas.shift(24),
        "llegadas_lag_168h": llegadas.shift(168),
        # Media movil de 7 dias sobre salidas, con shift de 24h para no filtrar
        "media_7d": salidas.shift(24).rolling(window=ROLLING_WINDOW_H, min_periods=1).mean(),
    }

    # Total de salidas del sistema en t-24 (mismo valor para todas las estaciones)
    salidas_sistema_lag_24h = salidas.sum(axis=1).shift(24)

    # --- 2. Stack a formato largo ---
    parts = [w.stack().rename(col) for col, w in wide.items()]
    df = pd.concat(parts, axis=1).reset_index()

    df["salidas_sistema_lag_24h"] = df["hora_inicio"].map(salidas_sistema_lag_24h)

    # --- 3. Features temporales ---
    df["hora"] = df["hora_inicio"].dt.hour.astype("int8")
    df["dia_semana"] = df["hora_inicio"].dt.day_of_week.astype("int8")
    df["mes"] = df["hora_inicio"].dt.month.astype("int8")
    df["es_fin_de_semana"] = df["dia_semana"] >= 5
    df["es_hora_pico_manana"] = df["hora"].isin(PEAK_HOURS_MORNING)
    df["es_hora_pico_tarde"] = df["hora"].isin(PEAK_HOURS_AFTERNOON)

    # --- 4. Atributos estaticos de la estacion ---
    st_cols = stations.set_index("estacion_id")[["capacidad", "lat", "lon"]]
    df = df.merge(st_cols, left_on="estacion_id", right_index=True, how="left")

    df = df.sort_values(["estacion_id", "hora_inicio"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "trips_v1.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "demand_features_v2.parquet",
    raw_dir: Path = RAW_DATA_DIR,
) -> None:
    """Genera el panel v2 (legacy) desde trips_v1.parquet."""
    logger.info(f"Leyendo {input_path} y catalogo de estaciones...")
    df = pd.read_parquet(input_path, columns=["inicio", "origen_id"]).dropna()
    stations = MiBiciLoader(raw_dir).load_stations()

    logger.info("Construyendo matriz de demanda horaria...")
    demand = hourly_demand(df)
    n_hours, n_stations = demand.shape
    logger.info(f"Demanda: {n_hours:,} horas x {n_stations} estaciones")

    logger.info("Calculando features de rezago, temporales y de estacion...")
    features = build_features(demand, stations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.success(
        f"Guardado {output_path} ({len(features):,} filas, {features.shape[1]} columnas)"
    )


@app.command(name="build-v3")
def build_v3(
    input_path: Path = INTERIM_DATA_DIR / "trips_v1.parquet",
    raw_dir: Path = RAW_DATA_DIR,
) -> None:
    """Genera el panel v3 (salidas + llegadas, lags >= 24h) desde trips_v1.parquet.

    La ruta de salida se lee de params.yaml (datos.features).
    """
    params = load_params()
    output_path: Path = params["datos"]["features"]

    logger.info(f"Leyendo {input_path} y catalogo de estaciones...")
    df = pd.read_parquet(
        input_path,
        columns=["inicio", "fin", "origen_id", "destino_id"],
    )
    stations = MiBiciLoader(raw_dir).load_stations()

    logger.info("Construyendo matrices de salidas y llegadas por hora...")
    df_sal = df.dropna(subset=["inicio", "origen_id"])
    salidas = hourly_demand(df_sal)

    df_arr = df.dropna(subset=["fin", "destino_id"])
    llegadas = hourly_arrivals(df_arr)

    n_hours = len(salidas.index.union(llegadas.index))
    n_stations = len(salidas.columns.union(llegadas.columns))
    logger.info(f"Rango: {n_hours:,} horas x {n_stations} estaciones")

    logger.info("Calculando features v3 (lags >= 24h, salidas y llegadas)...")
    features = build_features_v3(salidas, llegadas, stations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.success(
        f"Guardado {output_path} ({len(features):,} filas, {features.shape[1]} columnas)"
    )


if __name__ == "__main__":
    app()
