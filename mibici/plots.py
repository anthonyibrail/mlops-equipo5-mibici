"""EDA plots for the MiBici demand prediction project.

Generates 6 figures from the trips and features datasets:
  01_demanda_por_hora      -- Mean demand by hour of day (weekday vs weekend)
  02_heatmap_hora_dow      -- Mean demand heatmap: hour x day of week
  03_tendencia_mensual     -- Monthly total system demand trend
  04_top_estaciones        -- Top 10 stations by mean hourly demand
  05_duracion_viajes       -- Trip duration distribution (trips_v1)
  06_correlacion_features  -- Feature correlation with target (demanda)

All figures are saved to reports/figures/ and logged as artifacts in the
MLflow experiment "mibici-eda". Pearson correlations are also logged as
metrics so they appear in the MLflow run comparison view.

Usage:
    python -m mibici.plots
    python -m mibici.plots --features-path data/processed/demand_features_v2.parquet
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display required

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import pandas as pd
import typer

from mibici.config import FIGURES_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mibici.dataset import MiBiciLoader
from mibici.versioning import TRACKING_URI

app = typer.Typer()

EDA_EXPERIMENT = "mibici-eda"
CORR_SAMPLE = 200_000  # rows sampled for correlation (stable estimate, fast)
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Paleta común de las gráficas: cambiar un color aquí lo cambia en todas
COLOR_PRIMARY = "#2166ac"  # serie principal (azul)
COLOR_SECONDARY = "#d6604d"  # serie de contraste y valores negativos (rojo)
COLOR_ACCENT = "#f4a582"  # marcas secundarias (naranja claro)
COLOR_HIGHLIGHT = "#fdae61"  # sombreado de horas pico

# Numeric features considered for correlation with the target
NUMERIC_FEATURES = [
    "demanda_lag_1h",
    "demanda_lag_24h",
    "demanda_lag_168h",
    "media_7d",
    "demanda_sistema_lag_1h",
    "hora",
    "dia_semana",
    "mes",
    "es_fin_de_semana",
    "es_hora_pico_manana",
    "es_hora_pico_tarde",
    "capacidad",
    "lat",
    "lon",
]


def _setup_style() -> None:
    """Apply a consistent, clean style to all plots."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    """Guarda la figura en path, libera la memoria y registra el archivo en el log.

    Args:
        fig: Figura de matplotlib ya construida.
        path: Ruta del PNG de salida.

    Returns:
        La misma ruta, para encadenar con la lista de figuras generadas.
    """
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path.name}")
    return path


# ---------------------------------------------------------------------------
# Plot 1: Mean demand by hour of day
# ---------------------------------------------------------------------------


def plot_hourly_demand(df: pd.DataFrame, output_dir: Path) -> Path:
    """Line chart of mean demand by hour, weekday vs weekend.

    Args:
        df: Features DataFrame with columns 'hora', 'dia_semana', 'demanda'.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    weekdays = df[df["dia_semana"] < 5].groupby("hora")["demanda"].mean()
    weekends = df[df["dia_semana"] >= 5].groupby("hora")["demanda"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weekdays.index, weekdays.values, label="Weekday", lw=2, color=COLOR_PRIMARY)
    ax.plot(
        weekends.index,
        weekends.values,
        label="Weekend",
        lw=2,
        color=COLOR_SECONDARY,
        linestyle="--",
    )

    # Shade peak hours
    for lo, hi in [(7, 9), (17, 19)]:
        ax.axvspan(lo, hi, alpha=0.10, color=COLOR_HIGHLIGHT, label="_nolegend_")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean trips started")
    ax.set_title("Mean demand by hour of day")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()

    return _save_figure(fig, output_dir / "01_demanda_por_hora.png")


# ---------------------------------------------------------------------------
# Plot 2: Hour x day-of-week demand heatmap
# ---------------------------------------------------------------------------


def plot_dow_hour_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap of mean demand by hour (rows) and day of week (columns).

    Args:
        df: Features DataFrame with columns 'hora', 'dia_semana', 'demanda'.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    pivot = (
        df.groupby(["hora", "dia_semana"])["demanda"].mean().unstack("dia_semana")
    )  # shape: (24 hours, 7 days)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(False)  # grid lines clash with heatmap cells

    im = ax.imshow(
        pivot.values, aspect="auto", cmap="YlOrRd", origin="upper", interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, label="Mean trips started")

    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_LABELS)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels(range(0, 24, 2))
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Hour of day")
    ax.set_title("Mean demand: hour x day of week")

    return _save_figure(fig, output_dir / "02_heatmap_hora_dow.png")


# ---------------------------------------------------------------------------
# Plot 3: Monthly total demand trend
# ---------------------------------------------------------------------------


def plot_monthly_trend(df: pd.DataFrame, output_dir: Path) -> Path:
    """Line chart of total system demand per calendar month.

    Args:
        df: Features DataFrame with columns 'hora_inicio' (datetime), 'demanda'.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    monthly = df.groupby(df["hora_inicio"].dt.to_period("M"))["demanda"].sum().sort_index()
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values, marker="o", markersize=4, lw=2, color=COLOR_PRIMARY)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=COLOR_PRIMARY)

    ax.set_xlabel("Month")
    ax.set_ylabel("Total trips started")
    ax.set_title("Monthly total system demand")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x / 1_000_000:.1f}M" if x >= 1_000_000 else f"{x / 1000:.0f}K"
        )
    )
    fig.autofmt_xdate()

    return _save_figure(fig, output_dir / "03_tendencia_mensual.png")


# ---------------------------------------------------------------------------
# Plot 4: Top 10 stations by mean hourly demand
# ---------------------------------------------------------------------------


def plot_station_demand(df: pd.DataFrame, stations: pd.DataFrame, output_dir: Path) -> Path:
    """Horizontal bar chart of the top 10 stations by mean hourly demand.

    Args:
        df: Features DataFrame with columns 'estacion_id', 'demanda'.
        stations: Station catalog with columns 'estacion_id', 'nombre'.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    station_avg = df.groupby("estacion_id")["demanda"].mean().reset_index(name="demanda_media")
    station_avg = station_avg.merge(
        stations[["estacion_id", "nombre"]], on="estacion_id", how="left"
    )
    station_avg["nombre"] = station_avg["nombre"].fillna(station_avg["estacion_id"].astype(str))

    top10 = station_avg.nlargest(10, "demanda_media").sort_values("demanda_media")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top10["nombre"], top10["demanda_media"], color=COLOR_PRIMARY)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("Mean trips started per hour")
    ax.set_title("Top 10 stations by mean hourly demand")

    return _save_figure(fig, output_dir / "04_top_estaciones.png")


# ---------------------------------------------------------------------------
# Plot 5: Trip duration distribution
# ---------------------------------------------------------------------------


def plot_duration_hist(trips_df: pd.DataFrame, output_dir: Path) -> Path:
    """Histogram of trip durations from the cleaned trips dataset.

    Focuses on trips <= 60 min (covers >99% of trips after cleaning).

    Args:
        trips_df: DataFrame with column 'duracion_min'.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    d = trips_df["duracion_min"].dropna()
    d_focus = d[d <= 60]
    pct_shown = 100 * len(d_focus) / len(d)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(d_focus, bins=60, color=COLOR_PRIMARY, edgecolor="white", linewidth=0.3)

    # Mark median and 95th percentile
    p50 = float(d.median())
    p95 = float(d.quantile(0.95))
    ax.axvline(p50, color=COLOR_SECONDARY, lw=1.5, linestyle="--", label=f"Median: {p50:.1f} min")
    ax.axvline(p95, color=COLOR_ACCENT, lw=1.5, linestyle=":", label=f"p95: {p95:.1f} min")

    ax.set_xlabel("Trip duration (min)")
    ax.set_ylabel("Trips")
    ax.set_title(f"Trip duration distribution (showing <= 60 min, {pct_shown:.1f}% of trips)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()

    return _save_figure(fig, output_dir / "05_duracion_viajes.png")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def compute_correlation(
    df: pd.DataFrame, sample_n: int = CORR_SAMPLE, seed: int = 42
) -> pd.Series:
    """Pearson correlation of each numeric feature with the target 'demanda'.

    Samples the DataFrame first for performance. NaN rows (early periods
    with no lag history) are dropped before sampling.

    Args:
        df: Features DataFrame containing 'demanda' and NUMERIC_FEATURES.
        sample_n: Number of rows to sample.
        seed: Random seed for reproducibility.

    Returns:
        Series of correlations indexed by feature name, sorted by absolute
        value in descending order.
    """
    cols = ["demanda"] + [c for c in NUMERIC_FEATURES if c in df.columns]
    sample = df[cols].dropna().sample(min(sample_n, len(df)), random_state=seed)
    corr = (
        sample.corr()["demanda"]
        .drop("demanda")
        .dropna()  # features with zero variance have undefined (NaN) correlation
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    return corr


def plot_correlation_bar(corr: pd.Series, output_dir: Path) -> Path:
    """Horizontal bar chart of feature correlations with 'demanda'.

    Args:
        corr: Series returned by compute_correlation().
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    colors = [COLOR_PRIMARY if v >= 0 else COLOR_SECONDARY for v in corr.values]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson correlation with demanda")
    ax.set_title("Feature correlation with target (demanda)")
    ax.set_xlim(-1, 1)

    return _save_figure(fig, output_dir / "06_correlacion_features.png")


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_to_mlflow(figures: list[Path], corr: pd.Series, tracking_uri: str = TRACKING_URI) -> str:
    """Log all figures as artifacts and correlation values as metrics.

    Args:
        figures: List of saved figure paths.
        corr: Correlation Series from compute_correlation().
        tracking_uri: MLflow tracking URI.

    Returns:
        MLflow run ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EDA_EXPERIMENT)
    with mlflow.start_run(run_name="eda_v2") as run:
        mlflow.set_tag("dataset_version", "v2")
        for feat, val in corr.items():
            mlflow.log_metric(f"corr_{feat}_demanda", round(float(val), 4))
        for fig_path in figures:
            mlflow.log_artifact(str(fig_path), artifact_path="figures")
        logger.success(f"EDA registrado en MLflow (run {run.info.run_id})")
        return run.info.run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "demand_features_v2.parquet",
    trips_path: Path = INTERIM_DATA_DIR / "trips_v1.parquet",
    figures_dir: Path = FIGURES_DIR,
    raw_dir: Path = RAW_DATA_DIR,
) -> None:
    """Generate all EDA plots from the v2 features dataset and trips_v1."""
    _setup_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading {features_path}...")
    df = pd.read_parquet(features_path)

    logger.info(f"Reading {trips_path} (duracion_min only)...")
    trips = pd.read_parquet(trips_path, columns=["duracion_min"])

    logger.info("Loading station catalog...")
    stations = MiBiciLoader(raw_dir).load_stations()

    logger.info("Generating plots...")
    saved: list[Path] = []
    saved.append(plot_hourly_demand(df, figures_dir))
    saved.append(plot_dow_hour_heatmap(df, figures_dir))
    saved.append(plot_monthly_trend(df, figures_dir))
    saved.append(plot_station_demand(df, stations, figures_dir))
    saved.append(plot_duration_hist(trips, figures_dir))

    logger.info("Computing feature correlations (sample 200K rows)...")
    corr = compute_correlation(df)
    saved.append(plot_correlation_bar(corr, figures_dir))

    logger.info("Logging to MLflow...")
    _log_to_mlflow(saved, corr)

    logger.success(f"Done: {len(saved)} figures saved to {figures_dir}")


if __name__ == "__main__":
    app()
