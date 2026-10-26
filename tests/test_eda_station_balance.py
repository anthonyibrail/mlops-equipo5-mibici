"""Pruebas del análisis de concentración y balance con datos sintéticos."""

import pandas as pd
import pytest

from mibici.eda_station_balance import (
    analyze_station_balance,
    demand_concentration,
    peak_station_balance,
    save_balance_plot,
)


@pytest.fixture
def stations():
    return pd.DataFrame({"estacion_id": [1, 2, 3], "nombre": ["A", "B", "C"]})


@pytest.fixture
def trips():
    # Un viaje empieza fuera del pico pero llega durante el pico.
    # Otro viaje transcurre el sábado y debe excluirse del balance laborable.
    return pd.DataFrame({
        "origen_id": [1, 1, 2, 2, 3, 1],
        "destino_id": [2, 3, 1, 1, 1, 2],
        "inicio": pd.to_datetime([
            "2024-01-08 07:10", "2024-01-08 07:50", "2024-01-08 06:55",
            "2024-01-08 08:30", "2024-01-08 09:10", "2024-01-13 08:00",
        ]),
        "fin": pd.to_datetime([
            "2024-01-08 07:25", "2024-01-08 08:05", "2024-01-08 07:15",
            "2024-01-08 08:45", "2024-01-08 09:40", "2024-01-13 08:10",
        ]),
    })


def test_concentration_uses_ceiling_and_share(trips):
    result = demand_concentration(trips, 0.10)
    assert result["estaciones_activas"] == 3
    assert result["estaciones_top"] == 1
    assert result["salidas_top"] == 3
    assert result["participacion_top_pct"] == 50.0


def test_concentration_fraction_half(trips):
    result = demand_concentration(trips, 0.5)
    assert result["estaciones_top"] == 2
    assert result["salidas_top"] == 5


def test_concentration_empty_data():
    result = demand_concentration(pd.DataFrame({"origen_id": pd.Series(dtype="Int64")}))
    assert result["estaciones_activas"] == 0
    assert result["participacion_top_pct"] == 0.0


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_invalid_fraction_raises(fraction, trips):
    with pytest.raises(ValueError):
        demand_concentration(trips, fraction)


def test_flow_uses_departure_and_arrival_timestamps(trips, stations):
    result = peak_station_balance(trips, stations).set_index("estacion_id")
    # La salida a 06:55 no cuenta, pero su llegada a 07:15 sí.
    assert result.loc[2, "salidas"] == 1
    assert result.loc[2, "llegadas"] == 1
    assert result.loc[2, "balance_neto"] == 0
    assert result.loc[1, "salidas"] == 2
    assert result.loc[1, "llegadas"] == 3
    assert result.loc[1, "balance_neto"] == -1


def test_flow_excludes_weekend(trips, stations):
    result = peak_station_balance(trips, stations)
    assert result["salidas"].sum() == 4  # solo 4 salidas de lunes en pico
    assert result["llegadas"].sum() == 5  # 5 llegadas de lunes en pico


def test_flow_includes_destination_only_station(stations):
    df = pd.DataFrame({
        "inicio": pd.to_datetime(["2024-01-08 06:40"]),
        "fin": pd.to_datetime(["2024-01-08 07:05"]),
        "origen_id": [1], "destino_id": [3],
    })
    balance = peak_station_balance(df, stations)
    assert balance.loc[0, "estacion_id"] == 3
    assert balance.loc[0, "salidas"] == 0
    assert balance.loc[0, "llegadas"] == 1
    assert balance.loc[0, "balance_neto"] == -1


def test_flow_requires_columns(stations):
    with pytest.raises(ValueError, match="Faltan columnas"):
        peak_station_balance(pd.DataFrame({"inicio": []}), stations)


def test_flow_requires_valid_hours(trips, stations):
    with pytest.raises(ValueError, match="hours"):
        peak_station_balance(trips, stations, hours=(24,))


def test_chart_is_created(trips, stations, tmp_path):
    balance = peak_station_balance(trips, stations)
    out = save_balance_plot(balance, tmp_path / "balance.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_outputs_include_real_metrics(trips, stations, tmp_path):
    report_dir = tmp_path / "reports"
    figure_dir = report_dir / "figures"
    summary = analyze_station_balance(trips, stations, report_dir, figure_dir)
    assert summary["concentracion_salidas"]["participacion_top_pct"] == 50.0
    assert (report_dir / "station_balance_summary.json").exists()
    for period in ("matutino", "vespertino"):
        assert (report_dir / f"station_balance_{period}.csv").exists()
        assert (figure_dir / f"07_balance_neto_{period}.png").exists()
        assert f"balance_pico_{period}" in summary


def test_summary_does_not_label_negative_balance_as_departure(trips, stations, tmp_path):
    summary = analyze_station_balance(trips, stations, tmp_path, tmp_path / "figs")
    morning = summary["balance_pico_matutino"]
    assert all(row["balance_neto"] > 0 for row in morning["top_predominio_salidas"])
    assert all(row["balance_neto"] < 0 for row in morning["top_predominio_llegadas"])
