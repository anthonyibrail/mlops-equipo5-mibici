# mlops-equipo5-mibici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prediccion de demanda horaria por estacion del sistema MiBici (Guadalajara) aplicando MLOps

**Equipo 5 · TC5044 Operaciones de aprendizaje automático**

Datos: [MiBici, datos abiertos](https://mibici.net/en/open-data/) (enero 2024 a septiembre 2026, ~12.5 M viajes).
Análisis de requerimientos: [`docs/ml_canvas.md`](docs/ml_canvas.md).

## Índice

1. [Requisitos e instalación](#requisitos-e-instalación)
2. [Pruebas](#pruebas)
3. [Cómo reproducir (Fase 1)](#cómo-reproducir-fase-1)
4. [Cómo reproducir (Fase 2)](#cómo-reproducir-fase-2)
5. [Pipeline con DVC (Entrega Final)](#pipeline-con-dvc-entrega-final)
6. [Integración continua (CI)](#integración-continua-ci)
7. [Servicio del modelo (FastAPI)](#servicio-del-modelo-fastapi)
8. [Contenedor Docker](#contenedor-docker)
9. [Reproducibilidad en otro entorno](#reproducibilidad-en-otro-entorno)
10. [Monitoreo de data drift](#monitoreo-de-data-drift)
11. [Organización del proyecto](#organización-del-proyecto)

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

**Verificar la instalación:** ejecutar las pruebas, como se explica en [Pruebas](#pruebas).

## Pruebas

Todas las pruebas se ejecutan con un solo comando desde la raíz del repositorio:

```bash
pytest -q          # o bien: make test
```

Usan datos sintéticos, así que no requieren descargar el dataset ni tener MLflow corriendo.

| Archivo | Qué valida |
|---|---|
| `test_data.py` | Reglas de limpieza (v1) |
| `test_features.py` | Panel v2, rezagos y ausencia de fuga de información |
| `test_phase2_pipeline.py` | Panel v3 y Pipeline de entrenamiento con datos sintéticos |
| `test_validation.py` | Validación walk-forward, líneas base y métricas |
| `test_tune.py`, `test_register.py`, `test_promote.py` | Búsqueda de hiperparámetros, registro y promoción a `@champion` |
| `test_drift.py` | Escenarios de drift, PSI y reglas de alerta |
| `test_plots.py`, `test_eda_station_balance.py` | Gráficas y análisis exploratorio |

_Pendiente: prueba de integración de punta a punta (carga de datos → preprocesamiento → predicción → métricas)._

## Cómo reproducir (Fase 1)

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m mibici.dataset download                 # 1. descarga CSV mensuales a data/raw
python -m mibici.dataset build                    # 2. v0: crudo unificado (data/interim/trips_v0.parquet)
python -m mibici.cleaning                         # 3. v1: limpio + reports/cleaning_report_v1.json
python -m mibici.versioning v0                    # 4. registra versiones en MLflow
python -m mibici.versioning v1 --parent v0
python -m mibici.features main                    # 5. v2: panel de features (data/processed/demand_features_v2.parquet)
python -m mibici.versioning v2 --parent v1 --date-col hora_inicio
python -m mibici.modeling.baseline                # 6. experimento baseline v0 vs v1
pytest tests                                      # pruebas de carga, limpieza y features

mlflow ui --backend-store-uri sqlite:///mlflow.db # ver versiones y experimentos
```

Con `make` instalado: `make download`, `make data`, `make features`, `make baseline`, `make test`.
O bien `make pipeline` para ejecutar `data`, `features` y `baseline` en secuencia.

## Cómo reproducir (Fase 2)

Sobre la rama con el PR #5 integrado, y con Python 3.11 y las dependencias instaladas:

```bash
make download      # omitir si ya existen CSV y catálogo en data/raw
make data          # v0 y v1
make features-v3   # panel v3 de salidas/llegadas y registro en MLflow
make lineas-base   # walk-forward de 6 cortes para líneas base
make train         # HistGradientBoosting por objetivo, corte inicial configurable
make predict       # predicciones guardadas en data/processed/predicciones.parquet
make test          # pruebas con datos sintéticos (sin descargar 12.5 M viajes)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

También se puede ejecutar `make pipeline-train` después de descargar los CSV.
El panel v3 se genera a partir de `trips_v1.parquet` (no de v2); en MLflow su `parent_version` es `v1`.
El entrenamiento actual usa un único corte para las métricas de `train.py`; **no equivale** a
la comparación *walk-forward* de seis cortes, que debe evaluarse de forma separada
antes de proclamar un modelo ganador.

## Pipeline con DVC (Entrega Final)

El pipeline completo está definido en `dvc.yaml` con cuatro etapas reproducibles:
`data → features → train → predict`. DVC rastrea los artefactos intermedios y detecta
automáticamente qué etapas necesitan re-ejecutarse cuando cambian sus dependencias.

**Prerequisito:** tener los CSV descargados en `data/raw/` (`make download`).

```bash
# Ver el grafo de dependencias del pipeline
dvc dag

# Mostrar los parámetros rastreados desde params.yaml
dvc params show

# Revisar qué etapas están desactualizadas
dvc status

# Reproducir solo las etapas con cambios (o todo si no hay dvc.lock)
dvc repro

# Forzar la re-ejecución completa
dvc repro -f
```

**Con remote de Google Drive** (opcional; sube parquets y pkl para compartirlos
sin re-ejecutar el pipeline en cada máquina):

```bash
# Configurar una sola vez (ver docs/instrucciones_dvc_gdrive.md)
dvc remote add -d gdrive gdrive://<ID-DEL-FOLDER>
dvc push          # sube todos los artefactos al remote

# En otra máquina, después de clonar el repo
dvc pull          # descarga los artefactos sin re-ejecutar
```

Ver instrucciones completas y criterios de éxito en
[`docs/instrucciones_dvc_gdrive.md`](docs/instrucciones_dvc_gdrive.md).

## Integración continua (CI)

Cada push o PR contra `main`/`dev` dispara automáticamente el workflow
`.github/workflows/ci.yml` que ejecuta:

1. `ruff format --check` — verifica formato del código
2. `ruff check` — verifica estilo
3. `pytest tests/ -q` — corre las 120 pruebas (usan datos sintéticos, no requieren datos reales)

El workflow de despliegue continuo (`.github/workflows/cd.yml`) construye y sube la imagen
Docker a DockerHub. Estará activo una vez que el `Dockerfile` esté disponible; hasta entonces
solo se puede disparar manualmente desde GitHub Actions.

## Servicio del modelo (FastAPI)

_Pendiente: endpoint `POST /predict`, esquema de entrada y salida (Swagger en `/docs`) y
ruta del artefacto del modelo (`models:/mibici-demanda-salidas@champion` y
`models:/mibici-demanda-llegadas@champion`, con su número de versión)._

## Contenedor Docker

_Pendiente: `docker build -t ml-service:latest .`, `docker run -p 8000:8000 ml-service:latest`
y liga a la imagen publicada en DockerHub con sus tags versionados._

## Reproducibilidad en otro entorno

_Pendiente: ejecución del pipeline y del servicio en un entorno limpio y comparación de
métricas y salidas contra las de referencia._

## Monitoreo de data drift

`mibici/modeling/drift.py` simula cambios en los datos después del despliegue y comprueba
que el monitoreo los detecte. Toda la configuración (escenarios y umbrales) está en la
sección `drift` de `params.yaml`.

```bash
make drift                                       # modelo @champion del Model Registry
python -m mibici.modeling.drift --source local   # pipelines de models/*.pkl
```

**Referencia.** El primer mes posterior al entrenamiento (`validacion.primer_mes_prueba`),
sin alteraciones. Cada escenario se genera a partir de ella con una semilla fija:

| Escenario | Qué simula |
|---|---|
| `sin_drift` | Control: los mismos datos; no debe generar alertas |
| `aumento_demanda` | La demanda real sube 30 % y las features aún no lo reflejan |
| `error_pipeline_conteos` | Un error en el pipeline duplica los conteos de los rezagos |
| `feature_faltante` | Se pierde la mitad de `media_7d` y `salidas_sistema_lag_24h` |
| `temporada_vacaciones` | Diciembre: la demanda baja 40 % |

**Detección en dos niveles.**

- **Datos de entrada** (no requieren conocer la demanda real): PSI por columna
  (aviso ≥ 0.10, alerta ≥ 0.25) y aumento de valores nulos (alerta ≥ 5 %).
  Las columnas de calendario no se vigilan, porque cambian cada mes por diseño.
- **Desempeño** (cuando llega la demanda real, un día después): cambio en el sesgo
  (aviso ≥ 10 %, alerta ≥ 20 %), cambio en el MAE (aviso ≥ 10 %, alerta ≥ 25 %) y
  comparación contra la línea base de promedio histórico.

Cada escenario recibe un estado (`OK`, `AVISO` o `ALERTA`), los motivos y una acción
propuesta: revisar el pipeline de features, reentrenar con datos recientes o usar la
línea base mientras se reentrena.

**Salidas.** `reports/drift/resumen_drift.csv`, `reports/drift/psi_por_columna.csv`,
las gráficas `reports/figures/drift_psi.png`, `drift_sesgo.png` y `drift_mae.png`,
y un run por escenario en el experimento `mibici-monitoreo` de MLflow.

## Organización del proyecto

```
├── LICENSE
├── Makefile             <- Comandos del pipeline (download, data, features, plots, baseline,
│                           pipeline) y de calidad (test, lint, format)
├── README.md
├── data                 <- No se versiona en Git; se regenera con el pipeline
│   ├── external         <- Datos de terceros (reservado para festivos y clima)
│   ├── interim          <- v0 y v1 del dataset (trips_v0.parquet, trips_v1.parquet)
│   ├── processed        <- v2 y v3, paneles de modelado (demand_features_v2/v3.parquet)
│   └── raw              <- CSV mensuales y catálogo de estaciones, inmutables
│
├── docs                 <- ML Canvas del proyecto
├── models               <- Modelos entrenados (Fase 2)
├── notebooks            <- Notebooks; convención: número, iniciales y descripción
│                           (por ejemplo, 1.0-ep-uso-del-pipeline)
├── params.yaml          <- Configuración central: problema, validación, modelos, MLflow y drift
├── pyproject.toml       <- Metadatos del paquete y configuración de ruff
├── references           <- Diccionarios de datos y material de consulta
├── reports
│   ├── cleaning_report_v1.json  <- Filas afectadas y justificación de cada regla de limpieza
│   ├── model_selection  <- Resultados de la búsqueda y de la evaluación walk-forward
│   ├── drift            <- Resumen de la simulación de drift (make drift)
│   └── figures          <- Gráficas del EDA y del monitoreo de drift
├── requirements.txt     <- Dependencias con versiones fijadas
│
├── tests                <- Pruebas con pytest (ver la sección Pruebas)
│
└── mibici               <- Código fuente del pipeline
    ├── __init__.py
    ├── config.py        <- Rutas del proyecto, centralizadas
    ├── dataset.py       <- MiBiciDownloader y MiBiciLoader: descarga y unificación (v0)
    ├── cleaning.py      <- TripCleaner y Rule: reglas de limpieza (v1)
    ├── features.py      <- Panel estación × hora con variables temporales y rezagos (v2 y v3)
    ├── versioning.py    <- DatasetVersioner: registro de versiones en MLflow
    ├── plots.py         <- Gráficas del EDA, registradas en MLflow
    └── modeling
        ├── __init__.py
        ├── baseline.py   <- Baseline ingenuo estacional sobre las versiones (Fase 1)
        ├── validation.py <- Validación walk-forward, líneas base y métricas
        ├── train.py      <- Pipeline de scikit-learn y entrenamiento
        ├── tune.py       <- Búsqueda de hiperparámetros y evaluación de 6 cortes
        ├── register.py   <- Registro en el Model Registry (@candidate)
        ├── promote.py    <- Promoción a @champion si supera a la línea base
        ├── predict.py    <- Inferencia con el modelo @champion
        └── drift.py      <- Simulación de data drift y alertas
```

--------
