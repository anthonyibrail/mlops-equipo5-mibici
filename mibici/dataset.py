"""Ingesta de datos de MiBici: descarga de CSV mensuales y carga a un esquema único.

Uso:
    python -m mibici.dataset download --start 2024-01 --end 2026-09
    python -m mibici.dataset build            # data/raw -> data/interim/trips_v0.parquet
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests
import typer
from loguru import logger
from tqdm import tqdm

from mibici.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

OPEN_DATA_URL = "https://mibici.net/en/open-data/"
BASE_URL = "https://mibici.net"
STATIONS_FILE = "nomenclatura_2026_09.csv"

# Los encabezados cambian entre archivos (acentos y codificacion corruptos),
# pero el orden de las 8 columnas es siempre el mismo: se renombran por posicion.
TRIP_COLUMNS = [
    "viaje_id", "usuario_id", "genero", "anio_nacimiento",
    "inicio", "fin", "origen_id", "destino_id",
]
DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M")  # sep-oct 2024 vienen sin segundos


class MiBiciDownloader:
    """Descarga los CSV mensuales publicados en el portal de datos abiertos."""

    def __init__(self, out_dir: Path = RAW_DATA_DIR):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def list_files(self) -> dict[str, str]:
        """Devuelve {'AAAA_MM': url} con todos los meses publicados."""
        html = requests.get(OPEN_DATA_URL, timeout=60).text
        paths = set(re.findall(r'/site/assets/files/\d+/datos_abiertos_\d{4}_\d{2}[^"]*\.csv', html))
        return {re.search(r"(\d{4}_\d{2})", p).group(1): BASE_URL + p for p in paths}

    def download(self, start: str, end: str) -> list[Path]:
        start_k, end_k = start.replace("-", "_"), end.replace("-", "_")
        files = {k: u for k, u in self.list_files().items() if start_k <= k <= end_k}
        stations = self.list_stations_url()
        targets = [(f"datos_abiertos_{k}.csv", u) for k, u in sorted(files.items())]
        targets.append((STATIONS_FILE, stations))
        saved = []
        for name, url in tqdm(targets, desc="Descargando"):
            path = self.out_dir / name
            if not path.exists():
                path.write_bytes(requests.get(url, timeout=600).content)
            saved.append(path)
        logger.info(f"{len(saved)} archivos en {self.out_dir}")
        return saved

    @staticmethod
    def list_stations_url() -> str:
        html = requests.get(OPEN_DATA_URL, timeout=60).text
        return BASE_URL + re.search(r'/site/assets/files/\d+/nomenclatura_[^"]*\.csv', html).group(0)


class MiBiciLoader:
    """Lee los CSV crudos y los unifica en un DataFrame tipado (version v0 del dataset).

    v0 no elimina ninguna fila: solo corrige la forma (codificacion, nombres,
    formatos de fecha y tipos). Las decisiones de limpieza viven en cleaning.py.
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        try:
            path.read_bytes().decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    @staticmethod
    def _parse_dates(s: pd.Series) -> pd.Series:
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in DATE_FORMATS:
            missing = out.isna()
            out[missing] = pd.to_datetime(s[missing], format=fmt, errors="coerce")
        return out

    def read_month(self, path: Path) -> pd.DataFrame:
        enc = self._detect_encoding(path)
        df = pd.read_csv(path, encoding=enc, header=0, names=TRIP_COLUMNS,
                         dtype=str, keep_default_na=False)
        df["archivo"] = re.search(r"(\d{4}_\d{2})", path.name).group(1)
        df["fecha_con_segundos"] = df["inicio"].str.count(":") == 2
        return df

    def load_trips(self) -> pd.DataFrame:
        files = sorted(self.raw_dir.glob("datos_abiertos_*.csv"))
        if not files:
            raise FileNotFoundError(f"No hay CSV de viajes en {self.raw_dir}")
        df = pd.concat([self.read_month(f) for f in tqdm(files, desc="Leyendo")],
                       ignore_index=True)
        df["inicio"] = self._parse_dates(df["inicio"])
        df["fin"] = self._parse_dates(df["fin"])
        for col in ["viaje_id", "usuario_id", "anio_nacimiento", "origen_id", "destino_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df["genero"] = df["genero"].str.strip().replace({"": pd.NA, "NULL": pd.NA})
        df["archivo"] = df["archivo"].astype("category")
        logger.info(f"v0: {len(df):,} viajes de {len(files)} archivos")
        return df

    def load_stations(self) -> pd.DataFrame:
        st = pd.read_csv(self.raw_dir / STATIONS_FILE, encoding=self._detect_encoding(
            self.raw_dir / STATIONS_FILE))
        return st.rename(columns={"id": "estacion_id", "name": "nombre", "latitude": "lat",
                                  "longitude": "lon", "dpcapacity": "capacidad"})


@app.command()
def download(start: str = "2024-01", end: str = "2026-09"):
    """Descarga los meses [start, end] y el catalogo de estaciones a data/raw."""
    MiBiciDownloader().download(start, end)


@app.command()
def build(output_path: Path = INTERIM_DATA_DIR / "trips_v0.parquet"):
    """Unifica los CSV crudos en data/interim/trips_v0.parquet (sin eliminar filas)."""
    df = MiBiciLoader().load_trips()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Guardado {output_path}")


if __name__ == "__main__":
    app()
