# mlops-equipo5-mibici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prediccion de demanda horaria por estacion del sistema MiBici (Guadalajara) aplicando MLOps

**Equipo 5 · TC5044 Operaciones de aprendizaje automático**

Datos: [MiBici, datos abiertos](https://mibici.net/en/open-data/) (enero 2024 a septiembre 2026, ~12.5 M viajes).
Análisis de requerimientos: [`docs/ml_canvas.md`](docs/ml_canvas.md).

## Requisitos e instalación

- **Python 3.11.** El paquete exige esta versión (`requires-python = "~=3.11.0"`); con 3.12 o superior la instalación falla.
- **Git** y, opcionalmente, **make** para usar los atajos del `Makefile`.
- Unos 2 GB de disco y al menos 8 GB de RAM para procesar los 12.5 M de viajes.

Las versiones de todas las dependencias están fijadas en `requirements.txt` para que cada integrante trabaje con el mismo entorno.

**Linux o WSL (con [uv](https://docs.astral.sh/uv/))**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11          # descarga Python 3.11 si no está instalado
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Windows (con Python 3.11 instalado desde python.org)**

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Verificar la instalación**

```bash
make test        # o bien: python -m pytest tests
```

Las pruebas usan datos sintéticos, por lo que no requieren descargar el dataset.

## Cómo reproducir (Fase 1)

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m mibici.dataset download                 # 1. descarga CSV mensuales a data/raw
python -m mibici.dataset build                    # 2. v0: crudo unificado (data/interim/trips_v0.parquet)
python -m mibici.cleaning                         # 3. v1: limpio + reports/cleaning_report_v1.json
python -m mibici.versioning v0                    # 4. registra versiones en MLflow
python -m mibici.versioning v1 --parent v0
python -m mibici.features                         # 5. v2: panel de features (data/processed/demand_features_v2.parquet)
python -m mibici.versioning v2 --parent v1 --date-col hora_inicio
python -m mibici.modeling.baseline                # 6. experimento baseline v0 vs v1
pytest tests                                      # pruebas de carga, limpieza y features

mlflow ui --backend-store-uri sqlite:///mlflow.db # ver versiones y experimentos
```

Con `make` instalado: `make download`, `make data`, `make features`, `make baseline`, `make test`.
O bien `make pipeline` para ejecutar `data`, `features` y `baseline` en secuencia.

## Organización del proyecto

```
├── LICENSE
├── Makefile             <- Comandos del pipeline (download, data, features, plots, baseline,
│                           pipeline) y de calidad (test, lint, format)
├── README.md
├── data                 <- No se versiona en Git; se regenera con el pipeline
│   ├── external         <- Datos de terceros (reservado para festivos y clima)
│   ├── interim          <- v0 y v1 del dataset (trips_v0.parquet, trips_v1.parquet)
│   ├── processed        <- v2, panel de modelado (demand_features_v2.parquet)
│   └── raw              <- CSV mensuales y catálogo de estaciones, inmutables
│
├── docs                 <- ML Canvas del proyecto
├── models               <- Modelos entrenados (Fase 2)
├── notebooks            <- Notebooks; convención: número, iniciales y descripción
│                           (por ejemplo, 1.0-ep-uso-del-pipeline)
├── pyproject.toml       <- Metadatos del paquete y configuración de ruff
├── references           <- Diccionarios de datos y material de consulta
├── reports
│   ├── cleaning_report_v1.json  <- Filas afectadas y justificación de cada regla de limpieza
│   └── figures          <- Gráficas del EDA (generadas con make plots)
├── requirements.txt     <- Dependencias con versiones fijadas
│
├── tests
│   ├── test_data.py     <- Reglas de limpieza
│   ├── test_features.py <- Panel v2, rezagos y ausencia de fuga de información
│   └── test_plots.py    <- Generación de gráficas y correlaciones
│
└── mibici               <- Código fuente del pipeline
    ├── __init__.py
    ├── config.py        <- Rutas del proyecto, centralizadas
    ├── dataset.py       <- MiBiciDownloader y MiBiciLoader: descarga y unificación (v0)
    ├── cleaning.py      <- TripCleaner y Rule: reglas de limpieza (v1)
    ├── features.py      <- Panel estación × hora con variables temporales y rezagos (v2)
    ├── versioning.py    <- DatasetVersioner: registro de versiones en MLflow
    ├── plots.py         <- Gráficas del EDA, registradas en MLflow
    └── modeling
        ├── __init__.py
        ├── baseline.py  <- Baseline ingenuo estacional sobre las versiones
        ├── predict.py   <- Inferencia (Fase 2)
        └── train.py     <- Entrenamiento (Fase 2)
```

--------
