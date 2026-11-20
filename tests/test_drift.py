"""Pruebas de la simulacion de data drift (datos sinteticos, sin disco ni MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibici.config import load_params
from mibici.modeling.drift import (
    data_drift,
    evaluate_alerts,
    make_scenario,
    missing_increase,
    psi,
)

TARGETS = ["salidas", "llegadas"]
CRITERIA = ["salidas", "llegadas", "balance"]
THRESHOLDS = {
    "psi_aviso": 0.10,
    "psi_alerta": 0.25,
    "faltantes_alerta": 0.05,
    "sesgo_aviso": 0.10,
    "sesgo_alerta": 0.20,
    "mae_aviso": 1.10,
    "mae_alerta": 1.25,
    "margen_linea_base": 0.02,
}


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame(
        {
            "salidas": rng.poisson(3, n),
            "llegadas": rng.poisson(3, n),
            "salidas_lag_24h": rng.poisson(3, n).astype(float),
            "media_7d": rng.gamma(3, 1, n),
            "hora": rng.integers(0, 24, n),
            "mes": np.full(n, 4, dtype=np.int8),
        }
    )


# --- Escenarios -----------------------------------------------------------------
def test_factor_one_leaves_reference_unchanged(reference):
    out = make_scenario(reference, {"tipo": "demanda", "factor": 1.0}, TARGETS, seed=1)
    pd.testing.assert_frame_equal(out, reference)


def test_demand_scenario_scales_targets_but_not_features(reference):
    out = make_scenario(reference, {"tipo": "demanda", "factor": 1.3}, TARGETS, seed=1)
    ratio = out["salidas"].sum() / reference["salidas"].sum()
    assert ratio == pytest.approx(1.3, rel=0.02)
    assert (out["salidas"] >= 0).all() and out["salidas"].dtype.kind == "i"
    pd.testing.assert_series_equal(out["salidas_lag_24h"], reference["salidas_lag_24h"])


def test_feature_scenario_multiplies_only_listed_columns(reference):
    spec = {"tipo": "features", "columnas": ["salidas_lag_24h"], "factor": 2.0}
    out = make_scenario(reference, spec, TARGETS, seed=1)
    np.testing.assert_allclose(out["salidas_lag_24h"], reference["salidas_lag_24h"] * 2)
    pd.testing.assert_series_equal(out["salidas"], reference["salidas"])


def test_missing_scenario_removes_requested_fraction(reference):
    spec = {"tipo": "faltantes", "columnas": ["media_7d"], "fraccion": 0.5}
    out = make_scenario(reference, spec, TARGETS, seed=1)
    assert out["media_7d"].isna().mean() == pytest.approx(0.5, abs=0.03)
    assert reference["media_7d"].notna().all()  # la referencia no se modifica


def test_seasonal_scenario_changes_month_and_demand(reference):
    out = make_scenario(reference, {"tipo": "estacional", "factor": 0.6, "mes": 12}, TARGETS, 1)
    assert (out["mes"] == 12).all()
    assert out["llegadas"].sum() / reference["llegadas"].sum() == pytest.approx(0.6, rel=0.03)


def test_unknown_scenario_type_fails(reference):
    with pytest.raises(ValueError):
        make_scenario(reference, {"tipo": "otro"}, TARGETS, seed=1)


# --- Drift en datos -------------------------------------------------------------
def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)) < 0.01


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 20000), rng.normal(1, 1, 20000)) > 0.25


def test_psi_handles_discrete_and_constant_columns():
    hours = np.tile(np.arange(24), 100)
    assert psi(hours, hours) == pytest.approx(0.0, abs=1e-9)
    assert psi(np.full(100, 4), np.full(100, 4)) == pytest.approx(0.0, abs=1e-9)
    changed = psi(np.full(100, 4), np.full(100, 12))
    assert np.isfinite(changed) and changed > 0.25


def test_missing_increase_and_data_drift_table(reference):
    current = reference.copy()
    current.loc[:999, "media_7d"] = np.nan
    assert missing_increase(reference["media_7d"], current["media_7d"]) == pytest.approx(0.2)
    table = data_drift(reference, current, ["media_7d", "hora"], bins=10)
    assert list(table["columna"]) == ["media_7d", "hora"]
    assert table.loc[0, "aumento_nulos"] == pytest.approx(0.2)


# --- Reglas de alerta -------------------------------------------------------------
def _table(psi_value=0.01, missing=0.0):
    return pd.DataFrame(
        {"columna": ["a", "b"], "psi": [psi_value, 0.0], "aumento_nulos": [missing, 0.0]}
    )


def _perf(scale, bias=0.0):
    return {**{f"{c}_mae": 0.6 * scale for c in CRITERIA}, **{f"{t}_sesgo": bias for t in TARGETS}}


def _alerts(table, perf, ref, base):
    return evaluate_alerts(table, perf, ref, base, TARGETS, CRITERIA, THRESHOLDS)


def test_no_drift_is_ok():
    r = _alerts(_table(), _perf(1.0), _perf(1.0), _perf(2.0))
    assert r["estado"] == "OK"
    assert r["accion"].startswith("Sin accion")


def test_input_drift_points_to_feature_pipeline():
    r = _alerts(_table(psi_value=0.5), _perf(1.3), _perf(1.0), _perf(2.0))
    assert r["estado"] == "ALERTA"
    assert "pipeline de features" in r["accion"]


def test_missing_values_raise_an_alert():
    r = _alerts(_table(missing=0.5), _perf(1.0), _perf(1.0), _perf(2.0))
    assert r["estado"] == "ALERTA"


def test_performance_loss_without_input_drift_suggests_retraining():
    r = _alerts(_table(), _perf(1.3), _perf(1.0), _perf(2.0))
    assert r["estado"] == "ALERTA"
    assert r["accion"].startswith("Reentrenar")
    assert r["ratio_mae_salidas"] == pytest.approx(1.3)


def test_moderate_loss_is_a_warning():
    r = _alerts(_table(), _perf(1.15), _perf(1.0), _perf(2.0))
    assert r["estado"] == "AVISO"


def test_bias_detects_overprediction_even_if_mae_drops():
    # Temporada baja: el MAE baja con la demanda, pero el modelo sobreestima
    r = _alerts(_table(), _perf(0.9, bias=0.5), _perf(1.0), _perf(2.0))
    assert r["estado"] == "ALERTA"
    assert "sobreestima" in r["motivos"]
    assert r["cambio_sesgo_salidas"] == pytest.approx(0.5)


def test_small_gap_to_baseline_is_tolerated():
    r = _alerts(_table(), _perf(1.0), _perf(1.0), _perf(0.99))
    assert r["estado"] == "OK"


def test_model_worse_than_baseline_falls_back_to_baseline():
    r = _alerts(_table(), _perf(1.3), _perf(1.0), _perf(1.2))
    assert "linea base" in r["accion"]


def test_all_empty_columns_still_raise_a_missing_values_alert():
    table = pd.DataFrame(
        {"columna": ["a", "b"], "psi": [np.nan, np.nan], "aumento_nulos": [1.0, 1.0]}
    )
    r = _alerts(table, _perf(1.0), _perf(1.0), _perf(2.0))
    assert r["estado"] == "ALERTA"
    assert r["psi_max"] == 0.0


def test_project_params_define_valid_scenarios():
    cfg = load_params()["drift"]
    assert {s["tipo"] for s in cfg["escenarios"].values()} <= {
        "demanda",
        "features",
        "faltantes",
        "estacional",
    }
    assert cfg["umbrales"]["psi_aviso"] < cfg["umbrales"]["psi_alerta"]
    assert cfg["umbrales"]["mae_aviso"] < cfg["umbrales"]["mae_alerta"]
