"""EDA de concentración de salidas y balance de viajes por estación.

Complementa las gráficas descriptivas con métricas relevantes para redistribución:
- Participación de las salidas en el 10 % de estaciones más demandadas.
- Balance neto (salidas - llegadas) en horas pico matutinas y vespertinas.

Las llegadas se agrupan por la fecha/hora de FIN y las salidas por INICIO.
El balance de viajes no equivale al inventario real de bicicletas: no incorpora
ocupación inicial, movimientos de redistribución ni demanda insatisfecha.

Uso:
    python -m mibici.eda_station_balance
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer
from loguru import logger

from mibici.config import FIGURES_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()
PEAK_MORNING = (7, 8, 9)
PEAK_EVENING = (17, 18, 19)
COLOR_POSITIVE = "#2166ac"
COLOR_NEGATIVE = "#d6604d"


def demand_concentration(trips: pd.DataFrame, fraction: float = 0.10) -> dict:
    """Cuota de salidas de las estaciones de origen más utilizadas.

    El denominador para elegir el 10 % es el número de estaciones con al menos
    una salida en el período (no todas las estaciones del catálogo histórico).
    Se redondea hacia arriba el número de estaciones seleccionadas.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction debe estar en el intervalo (0, 1]")
    counts = trips["origen_id"].dropna().value_counts()
    n_stations = len(counts)
    n_top = ceil(n_stations * fraction)
    total = int(counts.sum())
    top = int(counts.nlargest(n_top).sum()) if n_top else 0
    return {
        "fraccion_estaciones": fraction,
        "estaciones_activas": n_stations,
        "estaciones_top": n_top,
        "salidas_totales": total,
        "salidas_top": top,
        "participacion_top_pct": round(100 * top / total, 3) if total else 0.0,
    }


def peak_station_balance(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    hours: tuple[int, ...] = PEAK_MORNING,
) -> pd.DataFrame:
    """Balance neto de eventos ocurridos en horas pico entre lunes y viernes.

    Las salidas usan ``inicio`` y ``origen_id``; las llegadas usan ``fin`` y
    ``destino_id``. Un viaje iniciado antes de las 7 pero finalizado a las 8
    cuenta como llegada, no como salida en esta ventana.

    Valores positivos: predominan salidas (posible necesidad de bicicletas).
    Valores negativos: predominan llegadas (posible necesidad de anclajes).
    """
    if not hours or any(not 0 <= h <= 23 for h in hours):
        raise ValueError("hours debe contener horas válidas entre 0 y 23")

    required = {"inicio", "fin", "origen_id", "destino_id"}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")

    starts = trips["inicio"]
    ends = trips["fin"]
    start_mask = starts.dt.dayofweek.lt(5) & starts.dt.hour.isin(hours)
    end_mask = ends.dt.dayofweek.lt(5) & ends.dt.hour.isin(hours)

    departures = trips.loc[start_mask, "origen_id"].dropna().value_counts()
    arrivals = trips.loc[end_mask, "destino_id"].dropna().value_counts()
    station_ids = departures.index.union(arrivals.index)
    balances = pd.DataFrame({
        "salidas": departures.reindex(station_ids, fill_value=0),
        "llegadas": arrivals.reindex(station_ids, fill_value=0),
    })
    balances.index.name = "estacion_id"
    balances = balances.astype(int).reset_index()
    balances["balance_neto"] = balances["salidas"] - balances["llegadas"]

    # El catálogo aporta etiquetas, pero no sustituye el historial de eventos.
    metadata = stations[["estacion_id", "nombre"]].drop_duplicates("estacion_id")
    balances = balances.merge(metadata, on="estacion_id", how="left", validate="one_to_one")
    return balances.sort_values("balance_neto", ascending=False).reset_index(drop=True)


def save_balance_plot(
    balance: pd.DataFrame, output: Path, n: int = 10, label: str = "07:00-09:59"
) -> Path:
    """Grafica las estaciones de mayor balance positivo y negativo."""
    if n < 1:
        raise ValueError("n debe ser >= 1")
    positives = balance[balance["balance_neto"] > 0].nlargest(n, "balance_neto")
    negatives = balance[balance["balance_neto"] < 0].nsmallest(n, "balance_neto")
    ranked = pd.concat([negatives, positives]).sort_values("balance_neto")

    fig, ax = plt.subplots(figsize=(11, 7))
    if ranked.empty:
        ax.text(0.5, 0.5, "Sin balances netos en el período", ha="center", va="center")
        ax.set_yticks([])
    else:
        labels = [
            f"{row.nombre if pd.notna(row.nombre) else 'Estación'} ({row.estacion_id})"
            for row in ranked.itertuples()
        ]
        values = ranked["balance_neto"].tolist()
        colors = [COLOR_NEGATIVE if v < 0 else COLOR_POSITIVE for v in values]
        ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Balance neto por estación · lunes a viernes, {label}")
    ax.set_xlabel("Salidas − llegadas (total de eventos en el período)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def analyze_station_balance(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    report_dir: Path,
    figures_dir: Path,
) -> dict:
    """Genera JSON, CSV y PNG reproducibles, sin inventar métricas del dataset."""
    concentration = demand_concentration(trips)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def records(df: pd.DataFrame) -> list[dict]:
        return [
            {
                "estacion_id": int(row.estacion_id),
                "nombre": str(row.nombre) if pd.notna(row.nombre) else "Sin nombre",
                "salidas": int(row.salidas),
                "llegadas": int(row.llegadas),
                "balance_neto": int(row.balance_neto),
            }
            for row in df.itertuples()
        ]

    summary = {
        "fuente": "trips_v1.parquet (viajes limpios)",
        "periodo_inicio": str(trips["inicio"].min()) if not trips.empty else None,
        "periodo_fin": str(trips["inicio"].max()) if not trips.empty else None,
        "criterio": (
            "Concentración sobre estaciones con >=1 salida; flujo de lunes a viernes, "
            "clasificando salidas por inicio y llegadas por fin"
        ),
        "concentracion_salidas": concentration,
        "limitacion": (
            "El balance de viajes no representa inventario ni saturación observada; "
            "no incluye reubicaciones operativas ni demanda insatisfecha."
        ),
        "salidas": {},
    }
    periods = {
        "matutino": (PEAK_MORNING, "07:00-09:59"),
        "vespertino": (PEAK_EVENING, "17:00-19:59"),
    }
    for period, (hours, label) in periods.items():
        balance = peak_station_balance(trips, stations, hours=hours)
        csv_path = report_dir / f"station_balance_{period}.csv"
        png_path = figures_dir / f"07_balance_neto_{period}.png"
        balance.to_csv(csv_path, index=False)
        save_balance_plot(balance, png_path, label=label)
        summary[f"balance_pico_{period}"] = {
            "ventana_horaria": label,
            "estaciones_evaluadas": len(balance),
            "top_predominio_salidas": records(
                balance[balance["balance_neto"] > 0].nlargest(5, "balance_neto")
            ),
            "top_predominio_llegadas": records(
                balance[balance["balance_neto"] < 0].nsmallest(5, "balance_neto")
            ),
        }
        summary["salidas"][period] = {"tabla_csv": str(csv_path), "grafica_png": str(png_path)}
    json_path = report_dir / "station_balance_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


@app.command()
def main(
    trips_path: Path = INTERIM_DATA_DIR / "trips_v1.parquet",
    report_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    raw_dir: Path = RAW_DATA_DIR,
) -> None:
    """Ejecuta el análisis con el dataset limpio y el catálogo de estaciones."""
    from mibici.dataset import MiBiciLoader

    trips = pd.read_parquet(trips_path, columns=["inicio", "fin", "origen_id", "destino_id"])
    stations = MiBiciLoader(raw_dir).load_stations()
    summary = analyze_station_balance(trips, stations, report_dir, figures_dir)
    logger.success(
        "Concentración top 10%: "
        f"{summary['concentracion_salidas']['participacion_top_pct']:.3f}%; "
        f"resumen: {report_dir / 'station_balance_summary.json'}"
    )


if __name__ == "__main__":
    app()
