# mlops-equipo5-mibici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prediccion de demanda horaria por estacion del sistema MiBici (Guadalajara) aplicando MLOps

**Equipo 5 · TC5044 Operaciones de aprendizaje automático**

Datos: [MiBici, datos abiertos](https://mibici.net/en/open-data/) (enero 2024 a septiembre 2026, ~12.5 M viajes).
Análisis de requerimientos: [`docs/ml_canvas.md`](docs/ml_canvas.md).

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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

