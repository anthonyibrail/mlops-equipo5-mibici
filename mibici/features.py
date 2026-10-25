"""Feature engineering: viajes limpios -> panel de demanda horaria con features.

Transforma trips_v1.parquet en un panel (hora_inicio x estacion_id) con:
  - Variable objetivo: demanda (viajes que inician en esa estacion en esa hora)
  - Features temporales: hora, dia_semana, mes, flags de pico y fin de semana
  - Features de rezago: demanda_lag_1h, demanda_lag_24h, demanda_lag_168h
  - Media movil de 7 dias (sin filtrar datos del periodo objetivo)
  - Demanda total del sistema rezagada 1h
  - Atributos estaticos de la estacion: capacidad, lat, lon

Regla de no-filtracion (data leakage): todos los rezagos y la media movil
usan shift >= 1 para que ningun feature incluya informacion del periodo t.

Uso:
    python -m mibici.features
    python -m mibici.features --input-path data/interim/trips_v1.parquet
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mibici.dataset import MiBiciLoader

app = typer.Typer()

PEAK_HOURS_MORNING = {7, 8, 9}
PEAK_HOURS_AFTERNOON = {17, 18, 19}
ROLLING_WINDOW_H = 7 * 24  # 168 horas = 7 dias


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


def build_features(demand: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye el panel de features a partir de la matriz de demanda horaria.

    Todas las operaciones de rezago se calculan en formato ancho antes de
    hacer el stack a largo, lo que es mas eficiente que hacer melt repetidamente.

    Args:
        demand: Matriz (DatetimeIndex x estacion_id) producida por hourly_demand().
                index.name debe ser 'hora_inicio' y columns.name 'estacion_id'.
        stations: DataFrame con columnas estacion_id, capacidad, lat, lon.

    Returns:
        DataFrame largo, una fila por (hora_inicio, estacion_id), con features
        y la columna 'demanda' como variable objetivo.
    """
    # --- 1. Calcular todos los features de rezago en formato ancho ---
    wide = {
        "demanda": demand,
        "demanda_lag_1h": demand.shift(1),
        "demanda_lag_24h": demand.shift(24),
        "demanda_lag_168h": demand.shift(168),
        "media_7d": demand.shift(1).rolling(window=ROLLING_WINDOW_H, min_periods=1).mean(),
    }

    # Demanda total del sistema en t-1 (igual para todas las estaciones en t)
    system_lag_1h = demand.sum(axis=1).shift(1)

    # --- 2. Convertir a formato largo mediante stack ---
    # Cada parte es una Series con MultiIndex (hora_inicio, estacion_id)
    parts = [wide_df.stack().rename(col) for col, wide_df in wide.items()]
    df = pd.concat(parts, axis=1).reset_index()
    # Columnas resultantes: hora_inicio, estacion_id, demanda, demanda_lag_1h, ...

    # Mapear demanda del sistema (broadcast a todas las estaciones de cada hora)
    df["demanda_sistema_lag_1h"] = df["hora_inicio"].map(system_lag_1h)

    # --- 3. Features temporales ---
    df["hora"] = df["hora_inicio"].dt.hour.astype("int8")
    df["dia_semana"] = df["hora_inicio"].dt.dayofweek.astype("int8")
    df["mes"] = df["hora_inicio"].dt.month.astype("int8")
    df["es_fin_de_semana"] = df["dia_semana"] >= 5
    df["es_hora_pico_manana"] = df["hora"].isin(PEAK_HOURS_MORNING)
    df["es_hora_pico_tarde"] = df["hora"].isin(PEAK_HOURS_AFTERNOON)

    # --- 4. Atributos estaticos de la estacion ---
    st_cols = stations.set_index("estacion_id")[["capacidad", "lat", "lon"]]
    df = df.merge(st_cols, left_on="estacion_id", right_index=True, how="left")

    # Ordenar por estacion y tiempo (conveniente para modelado secuencial)
    df = df.sort_values(["estacion_id", "hora_inicio"]).reset_index(drop=True)
    return df


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "trips_v1.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "demand_features_v2.parquet",
    raw_dir: Path = RAW_DATA_DIR,
) -> None:
    """Genera el panel de features de demanda horaria desde trips_v1.parquet."""
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


if __name__ == "__main__":
    app()
