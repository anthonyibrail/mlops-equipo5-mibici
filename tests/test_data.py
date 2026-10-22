"""Pruebas de la carga y limpieza de datos (datos sinteticos, sin descargar nada)."""

import pandas as pd
import pytest

from mibici.cleaning import TripCleaner
from mibici.dataset import MiBiciLoader


@pytest.fixture
def stations():
    # Dos estaciones a ~1.1 km de distancia
    return pd.DataFrame({"estacion_id": [1, 2], "nombre": ["A", "B"],
                         "lat": [20.670, 20.680], "lon": [-103.350, -103.350],
                         "capacidad": [15, 15]})


def trip(viaje_id, origen, destino, inicio, minutos, genero="M", anio=1995, segundos=True):
    t0 = pd.Timestamp(inicio)
    return {"viaje_id": viaje_id, "usuario_id": 10, "genero": genero, "anio_nacimiento": anio,
            "inicio": t0, "fin": t0 + pd.Timedelta(minutes=minutos), "origen_id": origen,
            "destino_id": destino, "archivo": "2025_01", "fecha_con_segundos": segundos}


def make_df(rows):
    df = pd.DataFrame(rows)
    for c in ["viaje_id", "usuario_id", "anio_nacimiento", "origen_id", "destino_id"]:
        df[c] = df[c].astype("Int64")
    return df


def test_parse_both_date_formats():
    s = pd.Series(["2024-01-01 08:15:30", "05/09/2024 8:15"])
    parsed = MiBiciLoader._parse_dates(s)
    assert parsed.notna().all()
    assert parsed[1] == pd.Timestamp("2024-09-05 08:15")


def test_valid_trip_is_kept(stations):
    df = make_df([trip(1, 1, 2, "2025-01-10 08:00:00", 8)])
    assert len(TripCleaner(stations).clean(df)) == 1


@pytest.mark.parametrize("bad_trip", [
    trip(2, 1, 1, "2025-01-10 08:00:00", 1),       # viaje fantasma
    trip(3, 1, 2, "2025-01-10 08:00:00", 0.5),     # 1.1 km en 30 s: velocidad imposible
    trip(4, 1, 2, "2025-01-10 08:00:00", 400),     # outlier de duracion
    trip(5, 1, 99, "2025-01-10 08:00:00", 8),      # estacion desconocida
])
def test_invalid_trips_are_removed(stations, bad_trip):
    assert len(TripCleaner(stations).clean(make_df([bad_trip]))) == 0


def test_duplicates_removed(stations):
    row = trip(1, 1, 2, "2025-01-10 08:00:00", 8)
    assert len(TripCleaner(stations).clean(make_df([row, row]))) == 1


def test_missing_demographics_do_not_remove_trip(stations):
    df = make_df([trip(1, 1, 2, "2025-01-10 08:00:00", 8, genero=pd.NA, anio=2023)])
    out = TripCleaner(stations).clean(df)
    assert len(out) == 1
    assert out.loc[0, "genero"] == "Desconocido"
    assert pd.isna(out.loc[0, "anio_nacimiento"])


def test_minute_resolution_gets_tolerance():
    # ~0.78 km registrados en "1 min": con segundos seria 47 km/h (se elimina);
    # sin segundos se da 1 min de tolerancia (~23 km/h) y se conserva.
    near = pd.DataFrame({"estacion_id": [1, 2], "nombre": ["A", "B"], "lat": [20.670, 20.677],
                         "lon": [-103.350, -103.350], "capacidad": [15, 15]})
    with_sec = make_df([trip(1, 1, 2, "2025-01-10 08:00:00", 1, segundos=True)])
    no_sec = make_df([trip(1, 1, 2, "2024-09-10 08:00:00", 1, segundos=False)])
    assert len(TripCleaner(near).clean(with_sec)) == 0
    assert len(TripCleaner(near).clean(no_sec)) == 1
