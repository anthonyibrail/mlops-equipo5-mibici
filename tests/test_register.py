"""Pruebas ligeras del registro; no entrenan millones de filas ni crean versiones."""
import numpy as np
import pandas as pd
import pytest

from mibici.modeling.register import training_boundaries, validate_preds, summarized_validation


def test_training_boundaries():
    start, end = training_boundaries("2026-10", 12)
    assert start == pd.Timestamp("2025-10-01")
    assert end == pd.Timestamp("2026-10-01")


def test_bad_training_window():
    with pytest.raises(ValueError):
        training_boundaries("2026-10", 0)
    with pytest.raises(ValueError):
        training_boundaries("2026-13", 12)


def test_prediction_guard():
    assert validate_preds([0, 1.25, 8.0]).tolist() == [0, 1.25, 8.0]
    for bad in ([float("inf")], [float("nan")], [-1.0], [1001]):
        with pytest.raises(ValueError):
            validate_preds(bad)


def test_summarized_validation():
    report = pd.DataFrame({
        "objetivo": ["salidas"] * 6,
        "mae": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "tipo_corte": ["seleccion"] * 2 + ["holdout"] * 4,
    })
    metrics = summarized_validation(report, "salidas")
    assert metrics["mae_seis_cortes"] == pytest.approx(0.75)
    assert metrics["mae_holdout_diagnostico"] == pytest.approx(0.85)
    with pytest.raises(ValueError):
        summarized_validation(report.iloc[:-1], "salidas")
