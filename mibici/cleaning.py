"""Limpieza de viajes de MiBici (v0 -> v1).

Cada regla es una funcion pequena con nombre, justificacion y una mascara de filas
a eliminar. TripCleaner las aplica en orden y deja un reporte con cuantas filas
quito cada una: ese reporte es la documentacion de cambios del dataset.

Criterio general: el objetivo del proyecto es predecir DEMANDA (viajes que inician
por estacion y hora). Se elimina un registro solo si no representa un viaje real
o no se puede ubicar en una estacion. Un viaje real con datos demograficos
faltantes SIGUE contando: esos campos se marcan como desconocidos.

Uso:
    python -m mibici.cleaning     # data/interim/trips_v0.parquet -> trips_v1.parquet
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import typer
from loguru import logger

from mibici.config import INTERIM_DATA_DIR, REPORTS_DIR
from mibici.dataset import MiBiciLoader

app = typer.Typer()

# Umbrales (documentados en el reporte de limpieza)
GHOST_TRIP_MAX_MIN = 2       # misma estacion origen/destino en < 2 min: desenganche cancelado
MAX_DURATION_MIN = 180       # outlier: > 3 h (el percentil 99.9 es ~58 min)
MAX_SPEED_KMH = 30           # velocidad media imposible para una bicicleta urbana
MIN_BIRTH_YEAR = 1930
MIN_AGE = 12


@dataclass
class Rule:
    name: str
    reason: str
    mask: Callable[[pd.DataFrame], pd.Series]   # True = fila a eliminar


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


class TripCleaner:
    def __init__(self, stations: pd.DataFrame):
        self.stations = stations.set_index("estacion_id")
        self.report: list[dict] = []

    # ---------- columnas auxiliares ----------
    def _add_helpers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
        lat, lon = self.stations["lat"], self.stations["lon"]
        df["distancia_km"] = haversine_km(df["origen_id"].map(lat), df["origen_id"].map(lon),
                                          df["destino_id"].map(lat), df["destino_id"].map(lon))
        return df

    @staticmethod
    def _speed_kmh(d: pd.DataFrame) -> pd.Series:
        # sep-oct 2024 no traen segundos: 1 min de tolerancia para no castigar redondeos.
        # Duracion 0 entre estaciones distintas da velocidad infinita (y se elimina).
        minutes = d["duracion_min"] + np.where(d["fecha_con_segundos"], 0, 1)
        with np.errstate(divide="ignore"):
            return d["distancia_km"] / (minutes / 60)

    # ---------- reglas de eliminacion ----------
    def rules(self) -> list[Rule]:
        known = set(self.stations.index)
        return [
            Rule("duplicados", "Registro repetido con el mismo viaje_id",
                 lambda d: d.duplicated(subset="viaje_id", keep="first")),
            Rule("fecha_invalida", "Inicio o fin vacio o no interpretable",
                 lambda d: d["inicio"].isna() | d["fin"].isna()),
            Rule("fin_antes_de_inicio", "Duracion negativa",
                 lambda d: d["duracion_min"] < 0),
            Rule("estacion_desconocida", "Origen o destino no existe en el catalogo de estaciones",
                 lambda d: ~d["origen_id"].isin(known) | ~d["destino_id"].isin(known)),
            Rule("viaje_fantasma",
                 f"Misma estacion de origen y destino en menos de {GHOST_TRIP_MAX_MIN} min "
                 "(bicicleta retirada y devuelta: no es un viaje)",
                 lambda d: (d["origen_id"] == d["destino_id"])
                 & (d["duracion_min"] < GHOST_TRIP_MAX_MIN)),
            Rule("velocidad_imposible",
                 f"Entre estaciones distintas con velocidad media > {MAX_SPEED_KMH} km/h "
                 "(incluye duracion 0)",
                 lambda d: (d["origen_id"] != d["destino_id"]) & (self._speed_kmh(d) > MAX_SPEED_KMH)),
            Rule("duracion_outlier", f"Duracion mayor a {MAX_DURATION_MIN} min",
                 lambda d: d["duracion_min"] > MAX_DURATION_MIN),
        ]

    # ---------- correcciones que NO eliminan filas ----------
    def _fix_demographics(self, df: pd.DataFrame, total: int) -> pd.DataFrame:
        age = df["inicio"].dt.year - df["anio_nacimiento"]
        bad_year = df["anio_nacimiento"].notna() & (
            (df["anio_nacimiento"] < MIN_BIRTH_YEAR) | (age < MIN_AGE))
        self._log("anio_nacimiento_invalido -> NA",
                  f"Fuera de [{MIN_BIRTH_YEAR}, anio_viaje - {MIN_AGE}]; el viaje se conserva",
                  int(bad_year.sum()), total, removes=False)
        df.loc[bad_year, "anio_nacimiento"] = pd.NA
        missing_gender = df["genero"].isna()
        self._log("genero_faltante -> 'Desconocido'", "NULL o vacio; el viaje se conserva",
                  int(missing_gender.sum()), total, removes=False)
        df["genero"] = df["genero"].fillna("Desconocido").astype("category")
        return df

    def _log(self, name, reason, n, total, removes=True):
        self.report.append({"regla": name, "justificacion": reason,
                            "filas_afectadas": n, "pct": round(100 * n / max(total, 1), 3),
                            "accion": "eliminar" if removes else "corregir"})
        logger.info(f"{name}: {n:,} filas ({100 * n / max(total, 1):.3f} %)")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self.report = []
        n0 = len(df)
        df = self._add_helpers(df)
        for rule in self.rules():
            drop = rule.mask(df).fillna(False).astype(bool)
            self._log(rule.name, rule.reason, int(drop.sum()), n0)
            df = df[~drop]
        df = self._fix_demographics(df.reset_index(drop=True), n0)
        logger.success(f"v1: {len(df):,} viajes ({n0 - len(df):,} eliminados, "
                       f"{100 * (n0 - len(df)) / max(n0, 1):.2f} %)")
        return df

    def save_report(self, path: Path, n_in: int, n_out: int) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"filas_v0": n_in, "filas_v1": n_out,
                                    "reglas": self.report}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path


@app.command()
def main(input_path: Path = INTERIM_DATA_DIR / "trips_v0.parquet",
         output_path: Path = INTERIM_DATA_DIR / "trips_v1.parquet",
         report_path: Path = REPORTS_DIR / "cleaning_report_v1.json"):
    df = pd.read_parquet(input_path)
    cleaner = TripCleaner(MiBiciLoader().load_stations())
    clean = cleaner.clean(df)
    clean.to_parquet(output_path, index=False)
    cleaner.save_report(report_path, len(df), len(clean))
    logger.success(f"Guardados {output_path} y {report_path}")


if __name__ == "__main__":
    app()
