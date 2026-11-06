"""Pruebas del esquema de evaluacion (datos sinteticos, sin disco ni MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mibici.config import load_params
from mibici.modeling.validation import (
    HistoricalMean,
    SeasonalNaive,
    WalkForwardSplit,
    baseline_estimators,
    cross_validate_model,
    net_balance_metrics,
    regression_metrics,
    summarize,
)

PEAK = [7, 8, 9, 17, 18, 19]


@pytest.fixture
def panel() -> pd.DataFrame:
    """Panel horario de 2 estaciones de enero 2025 a junio 2025."""
    hours = pd.date_range("2025-01-01", "2025-06-30 23:00", freq="h")
    rng = np.random.default_rng(0)
    frames = []
    for st in (1, 2):
        base = 2 + st + 3 * np.isin(hours.hour, PEAK)
        sal = rng.poisson(base).astype(float)
        lle = rng.poisson(base).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "hora_inicio": hours,
                    "estacion_id": st,
                    "hora": hours.hour,
                    "dia_semana": hours.day_of_week,
                    "salidas": sal,
                    "llegadas": lle,
                    "salidas_lag_168h": pd.Series(sal).shift(168).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def splitter() -> WalkForwardSplit:
    return WalkForwardSplit("hora_inicio", "2025-04", n_splits=3, train_months=2)


# --- WalkForwardSplit ---------------------------------------------------------
def test_folds_boundaries(splitter):
    f = splitter.folds()
    assert list(f["inicio_prueba"].dt.strftime("%Y-%m")) == ["2025-04", "2025-05", "2025-06"]
    assert (f["inicio_entrenamiento"] == f["inicio_prueba"] - pd.DateOffset(months=2)).all()


def test_split_never_trains_on_test_or_future(panel, splitter):
    t = panel["hora_inicio"]
    for train_idx, test_idx in splitter.split(panel):
        assert len(set(train_idx) & set(test_idx)) == 0
        assert t.iloc[train_idx].max() < t.iloc[test_idx].min()
        assert t.iloc[test_idx].dt.month.nunique() == 1


def test_expanding_window_uses_all_history(panel):
    s = WalkForwardSplit("hora_inicio", "2025-04", n_splits=1, train_months=None)
    train_idx, _ = next(s.split(panel))
    assert panel["hora_inicio"].iloc[train_idx].min() == panel["hora_inicio"].min()


def test_split_works_as_sklearn_cv(panel, splitter):
    from sklearn.model_selection import cross_val_score

    X = panel[["hora_inicio", "hora"]]
    scores = cross_val_score(
        LinearRegression(), X[["hora"]], panel["salidas"], cv=list(splitter.split(X))
    )
    assert len(scores) == 3


def test_from_params_reads_project_config():
    s = WalkForwardSplit.from_params(load_params())
    assert s.get_n_splits() == len(s.folds())


# --- Lineas base ----------------------------------------------------------------
def test_seasonal_naive_returns_lag(panel):
    pred = SeasonalNaive("salidas_lag_168h").fit(panel).predict(panel)
    expected = panel["salidas_lag_168h"].fillna(0).to_numpy()
    np.testing.assert_array_equal(pred, expected)


def test_historical_mean_by_station_day_hour():
    X = pd.DataFrame({"estacion_id": [1, 1, 2], "dia_semana": [0, 0, 0], "hora": [8, 8, 8]})
    m = HistoricalMean().fit(X, [2.0, 4.0, 10.0])
    unseen = pd.DataFrame({"estacion_id": [1, 2, 3], "dia_semana": [0, 0, 0], "hora": [8, 8, 8]})
    np.testing.assert_allclose(m.predict(unseen), [3.0, 10.0, 16 / 3])


def test_baselines_use_target_specific_columns():
    est = baseline_estimators(load_params(), "llegadas")
    assert est["ingenuo_estacional"].required_columns() == ["llegadas_lag_168h"]


# --- Metricas -------------------------------------------------------------------
def test_regression_metrics_values():
    m = regression_metrics([0, 2, 4], [1, 2, 2], hours=[8, 12, 8], peak_hours=PEAK)
    assert m["mae"] == pytest.approx(1.0)
    assert m["mae_pico"] == pytest.approx(1.5)
    assert m["mae_relativo"] == pytest.approx(0.5)


def test_net_balance_metrics():
    m = net_balance_metrics([5, 1], [1, 5], [4, 1], [1, 4], hours=[8, 18], peak_hours=PEAK)
    assert m["balance_mae"] == pytest.approx(1.0)


# --- Evaluacion por cortes --------------------------------------------------------
def test_cross_validate_model_returns_one_row_per_fold(panel, splitter):
    folds, preds = cross_validate_model(
        HistoricalMean(),
        panel,
        "salidas",
        ["estacion_id", "dia_semana", "hora"],
        splitter,
        PEAK,
        return_predictions=True,
    )
    assert list(folds["mes_prueba"]) == ["2025-04", "2025-05", "2025-06"]
    assert len(preds) == folds["filas_prueba"].sum()
    s = summarize(folds)
    assert s["mae_promedio"] == pytest.approx(folds["mae"].mean())
