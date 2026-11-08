"""Pruebas rápidas de integración de la Fase 2 con datos sintéticos."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from mibici.features import app, build_features_v3, hourly_arrivals
from mibici.modeling.train import ALL_FEATURES, build_pipeline, temporal_split
from mibici.versioning import VERSIONS


def synthetic_panel() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=200, freq="h", name="hora_inicio")
    cols = pd.Index([101, 102], name="estacion_id")
    rng = np.random.default_rng(42)
    salidas = pd.DataFrame(rng.poisson(3, (len(hours), 2)), index=hours, columns=cols)
    llegadas = pd.DataFrame(rng.poisson(2, (len(hours), 2)), index=hours, columns=cols)
    stations = pd.DataFrame(
        {
            "estacion_id": [101, 102],
            "capacidad": [20, 30],
            "lat": [20.68, 20.70],
            "lon": [-103.35, -103.36],
        }
    )
    return build_features_v3(salidas, llegadas, stations)


def test_v3_is_known_in_dataset_registry():
    assert VERSIONS["v3"][0] == "demand_features_v3.parquet"


def test_features_cli_lists_both_versions():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "main" in result.output
    assert "build-v3" in result.output


def test_hourly_arrivals_counts_and_zero_fills():
    frame = pd.DataFrame(
        {
            "fin": [pd.Timestamp("2026-01-01 07:04"), pd.Timestamp("2026-01-01 09:20")],
            "destino_id": [101, 101],
        }
    )
    result = hourly_arrivals(frame)
    assert result.loc[pd.Timestamp("2026-01-01 07:00"), 101] == 1
    assert result.loc[pd.Timestamp("2026-01-01 08:00"), 101] == 0
    assert result.loc[pd.Timestamp("2026-01-01 09:00"), 101] == 1


def test_v3_has_two_targets_and_lags_from_the_past():
    df = synthetic_panel()
    assert len(df) == 200 * 2
    assert {"salidas", "llegadas", *ALL_FEATURES}.issubset(df.columns)
    st = df[df["estacion_id"] == 101].sort_values("hora_inicio").reset_index(drop=True)
    assert pd.isna(st.loc[0, "salidas_lag_24h"])
    assert st.loc[50, "salidas_lag_24h"] == st.loc[26, "salidas"]
    assert st.loc[50, "llegadas_lag_24h"] == st.loc[26, "llegadas"]
    assert pd.isna(st.loc[50, "salidas_lag_168h"])


def test_temporal_split_never_leaks_future_rows():
    frame = pd.DataFrame(
        {"hora_inicio": pd.to_datetime(["2026-03-31 23:00", "2026-04-01 00:00"])}
    )
    train, test = temporal_split(frame, cutoff="2026-04")
    assert len(train) == len(test) == 1
    assert train["hora_inicio"].max() < test["hora_inicio"].min()


def test_pipeline_train_predict_and_pickle_roundtrip():
    df = synthetic_panel().iloc[169:].copy()
    config = {"loss": "poisson", "max_iter": 6, "max_leaf_nodes": 7,
              "min_samples_leaf": 2, "random_state": 42}
    model = build_pipeline(config)
    X = df[ALL_FEATURES]
    y = df["salidas"].astype(float)
    model.fit(X, y)
    expected = model.predict(X)
    restored = pickle.loads(pickle.dumps(model))
    actual = restored.predict(X)
    np.testing.assert_allclose(expected, actual)
    assert np.isfinite(actual).all()
    assert (actual >= 0).all()
