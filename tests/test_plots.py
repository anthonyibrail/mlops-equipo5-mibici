"""Tests for plots.py: verify each figure is created without errors.

All tests use synthetic DataFrames in memory; no disk data or MLflow required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibici.plots import (
    NUMERIC_FEATURES,
    compute_correlation,
    plot_correlation_bar,
    plot_dow_hour_heatmap,
    plot_duration_hist,
    plot_hourly_demand,
    plot_monthly_trend,
    plot_station_demand,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 504  # 3 weeks * 7 days * 24 hours = 504 unique timestamps (nice for dow coverage)


@pytest.fixture
def feat_df() -> pd.DataFrame:
    """Synthetic features DataFrame covering 3 full weeks for 3 stations."""
    idx = pd.date_range("2024-01-08", periods=N_ROWS, freq="h")  # starts on Monday
    rng = np.random.default_rng(seed=0)
    station_ids = [1, 2, 3]

    frames = []
    for sid in station_ids:
        hours = idx.hour
        df_s = pd.DataFrame({
            "hora_inicio":            idx,
            "estacion_id":            sid,
            "demanda":                rng.integers(0, 10, size=N_ROWS).astype(float),
            "demanda_lag_1h":         rng.integers(0, 10, size=N_ROWS).astype(float),
            "demanda_lag_24h":        rng.integers(0, 10, size=N_ROWS).astype(float),
            "demanda_lag_168h":       rng.integers(0, 10, size=N_ROWS).astype(float),
            "media_7d":               rng.integers(0, 10, size=N_ROWS).astype(float),
            "demanda_sistema_lag_1h": rng.integers(0, 50, size=N_ROWS).astype(float),
            "hora":                   hours.astype("int8"),
            "dia_semana":             idx.dayofweek.astype("int8"),
            "mes":                    idx.month.astype("int8"),
            "es_fin_de_semana":       idx.dayofweek >= 5,
            "es_hora_pico_manana":    pd.Series(hours).isin([7, 8, 9]).values,
            "es_hora_pico_tarde":     pd.Series(hours).isin([17, 18, 19]).values,
            "capacidad":              15 * sid,
            "lat":                    20.670 + sid * 0.01,
            "lon":                    -103.350 - sid * 0.01,
        })
        frames.append(df_s)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def trips_df() -> pd.DataFrame:
    """Synthetic trip durations (exponential distribution, trimmed at 180 min)."""
    rng = np.random.default_rng(seed=0)
    durations = rng.exponential(scale=15, size=2000).clip(0, 180)
    return pd.DataFrame({"duracion_min": durations})


@pytest.fixture
def stations_df() -> pd.DataFrame:
    """Minimal station catalog for 3 stations."""
    return pd.DataFrame({
        "estacion_id": [1, 2, 3],
        "nombre":      ["Sta A", "Sta B", "Sta C"],
        "lat":         [20.670, 20.680, 20.690],
        "lon":         [-103.350, -103.360, -103.370],
        "capacidad":   [15, 20, 10],
    })


# ---------------------------------------------------------------------------
# Tests for individual plot functions
# ---------------------------------------------------------------------------

def test_plot_hourly_demand_creates_file(feat_df, tmp_path):
    path = plot_hourly_demand(feat_df, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.suffix == ".png"


def test_plot_dow_hour_heatmap_creates_file(feat_df, tmp_path):
    path = plot_dow_hour_heatmap(feat_df, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.suffix == ".png"


def test_plot_monthly_trend_creates_file(feat_df, tmp_path):
    path = plot_monthly_trend(feat_df, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_station_demand_creates_file(feat_df, stations_df, tmp_path):
    path = plot_station_demand(feat_df, stations_df, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_duration_hist_creates_file(trips_df, tmp_path):
    path = plot_duration_hist(trips_df, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_correlation_bar_creates_file(feat_df, tmp_path):
    corr = compute_correlation(feat_df, sample_n=500)
    path = plot_correlation_bar(corr, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests for compute_correlation
# ---------------------------------------------------------------------------

def test_correlation_excludes_target(feat_df):
    """The returned Series must not contain 'demanda' itself."""
    corr = compute_correlation(feat_df, sample_n=500)
    assert "demanda" not in corr.index


def test_correlation_contains_expected_features(feat_df):
    """Features with non-constant values in the sample appear in the output."""
    corr = compute_correlation(feat_df, sample_n=500)
    # These features always have variance in the fixture (random values)
    always_varying = [
        "demanda_lag_1h", "demanda_lag_24h", "demanda_lag_168h",
        "hora", "dia_semana",
    ]
    missing = [f for f in always_varying if f not in corr.index]
    assert not missing, f"Features missing from correlation: {missing}"


def test_correlation_values_in_range(feat_df):
    """All returned values are within [-1, 1] and non-NaN.

    compute_correlation drops constant features (undefined correlation),
    so NaN must not appear in the output.
    """
    corr = compute_correlation(feat_df, sample_n=500)
    assert corr.notna().all()
    assert (corr.abs() <= 1.0).all()


def test_correlation_sorted_by_absolute_value(feat_df):
    """Output is sorted by absolute value, descending."""
    corr = compute_correlation(feat_df, sample_n=500)
    abs_vals = list(corr.abs().values)
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_correlation_reproducible(feat_df):
    """Same seed produces the same result."""
    c1 = compute_correlation(feat_df, sample_n=500, seed=7)
    c2 = compute_correlation(feat_df, sample_n=500, seed=7)
    pd.testing.assert_series_equal(c1, c2)
