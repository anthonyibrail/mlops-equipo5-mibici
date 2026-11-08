#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-equipo5-mibici
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Descargar datos de MiBici (ene-2024 a sep-2026) a data/raw
.PHONY: download
download:
	$(PYTHON_INTERPRETER) -m mibici.dataset download --start 2024-01 --end 2026-09

## Pipeline de datos: v0 (crudo unificado) -> v1 (limpio) y registro en MLflow
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset build
	$(PYTHON_INTERPRETER) -m mibici.cleaning
	$(PYTHON_INTERPRETER) -m mibici.versioning v0
	$(PYTHON_INTERPRETER) -m mibici.versioning v1 --parent v0

## Experimento baseline sobre las versiones del dataset
.PHONY: baseline
baseline:
	$(PYTHON_INTERPRETER) -m mibici.modeling.baseline

## Feature engineering: trips_v1 -> demand_features_v2 + registro MLflow v2
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.features main
	$(PYTHON_INTERPRETER) -m mibici.versioning v2 --parent v1 --date-col hora_inicio

## Generar plots de EDA y registrar en MLflow
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m mibici.plots

## Lineas base con validacion movil (params.yaml) y registro en MLflow
.PHONY: lineas-base
lineas-base:
	$(PYTHON_INTERPRETER) -m mibici.modeling.validation

## Panel v3: salidas y llegadas separadas, lags >= 24h (Fase 2)
.PHONY: features-v3
features-v3:
	$(PYTHON_INTERPRETER) -m mibici.features build-v3
	$(PYTHON_INTERPRETER) -m mibici.versioning v3 --parent v1 --date-col hora_inicio

## Entrenamiento de pipelines por objetivo y registro en MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train

## Busqueda de hiperparametros (opcional: reescribe tuning_winners.json)
.PHONY: tune
tune:
	$(PYTHON_INTERPRETER) -m mibici.modeling.tune search

## Evaluacion walk-forward de 6 cortes con los ganadores versionados
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) -m mibici.modeling.tune evaluate

## Refit final y registro en el Model Registry con alias @candidate
.PHONY: register
register:
	$(PYTHON_INTERPRETER) -m mibici.modeling.register

## Promocion a @champion si supera a la linea base (salidas, llegadas y balance)
.PHONY: promote
promote:
	$(PYTHON_INTERPRETER) -m mibici.modeling.promote

## Predicciones con el modelo @champion del Model Registry
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m mibici.modeling.predict

## Pipeline completo: descarga, datos, features, baseline y plots
.PHONY: pipeline
pipeline: data features baseline plots

## Pipeline de entrenamiento (Fase 2): datos, features v3, lineas base, train, predict
.PHONY: pipeline-train
pipeline-train: data features-v3 lineas-base train evaluate register promote predict


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
