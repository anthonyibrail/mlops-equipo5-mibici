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
	$(PYTHON_INTERPRETER) -m mibici.features
	$(PYTHON_INTERPRETER) -m mibici.versioning v2 --parent v1 --date-col hora_inicio

## Pipeline completo: descarga, datos, features y baseline
.PHONY: pipeline
pipeline: data features baseline


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
