# PR: feat(api + dvc + ci) — Servicio FastAPI, pipeline DVC y CI/CD

**Branch:** `feature/fastapi` → `main`

---

## Que hace este PR

Este PR consolida la **Entrega Final** de MLOps en cuatro areas:

1. **Servicio FastAPI** — API REST de pronostico de demanda (3 endpoints)
2. **Pipeline DVC reproducible** — 5 stages con tracking de parametros y metricas
3. **CI/CD con GitHub Actions** — lint + tests en cada PR; Docker build preparado
4. **Remote de artefactos** — configuracion de Google Drive para DVC

---

## 1. Servicio FastAPI

### Archivos

| Archivo | Descripcion |
|---|---|
| `mibici/api/schemas.py` | Schemas Pydantic: 16 features de entrada, response con salidas/llegadas/balance |
| `mibici/api/loader.py` | Carga los pkl de `models/` al arrancar; bool_to_int para pickle estable |
| `mibici/api/main.py` | Lifespan + endpoints /health, /predict, /metrics |
| `mibici/api/__init__.py` | Paquete |
| `tests/test_api.py` | 15 tests con pipelines mockeados (0 skipped) |
| `Makefile` | Target `api` para arrancar uvicorn |
| `requirements.txt` | fastapi==0.159.1, uvicorn==0.54.0, httpx==0.28.1 |

### Endpoints

**GET /health**
```json
{"status": "ok", "source": "local", "targets": ["salidas", "llegadas"]}
```

**POST /predict** — 16 features del panel v3 (lags opcionales → NaN nativo)
```json
// Request
{"hora": 8, "dia_semana": 1, "mes": 4,
 "es_fin_de_semana": false, "es_hora_pico_manana": true, "es_hora_pico_tarde": false,
 "capacidad": 15.0, "lat": 20.6736, "lon": -103.3435}

// Response
{"salidas_pred": 4.1, "llegadas_pred": 2.8, "balance": 1.3}
```

**GET /metrics** — lee `walk_forward_6_cortes.csv`
```json
{"mae_salidas": 0.674, "mae_llegadas": 0.662, "mae_balance": 0.716,
 "n_folds": 6, "source": "reports/model_selection/walk_forward_6_cortes.csv"}
```

### Tests
```
pytest tests/test_api.py   → 15 passed
pytest tests/ -q           → 142 passed (103 original + 24 DVC + 15 API)
```

### Decisiones tecnicas

- **Lifespan**: `@asynccontextmanager` carga los modelos una vez en `app.state.pipelines`
- **bool_to_int**: en `mibici/modeling/__init__.py` para path de pickle estable
- **Metricas**: usa `REPORTS_DIR` (path absoluto) para funcionar en cualquier directorio
- **Tests mockeados**: parche en atributo del modulo para que el lifespan use el mock

---

## 2. Pipeline DVC reproducible

### Archivos

| Archivo | Descripcion |
|---|---|
| `dvc.yaml` | Pipeline de 5 stages reproducibles |
| `dvc.lock` | Estado actual de artefactos (hashes) |
| `.dvc/config` | Remote Google Drive con service account |
| `.dvcignore` | Patrones ignorados por DVC |
| `requirements.txt` | dvc==3.67.1, dvc-gdrive==3.0.1 |
| `tests/test_dvc_pipeline.py` | 24 tests de estructura del pipeline |
| `docs/instrucciones_dvc_gdrive.md` | Instrucciones + troubleshooting |

### Stages del pipeline

```
data → features → train → predict
                        ↘ drift
```

| Stage | Que hace | Outputs clave |
|---|---|---|
| `data` | Ingesta + limpieza + versionado v0/v1 | `data/interim/trips_v*.parquet` |
| `features` | Panel v3 sin fuga de informacion | `data/processed/demand_features_v3.parquet` |
| `train` | Lineas base + train + walk-forward + register + promote | `models/pipeline_*.pkl`, CSV metricas |
| `predict` | Inferencia con modelos locales | `data/processed/predicciones.parquet` |
| `drift` | Simulacion de data drift (5 escenarios) | `reports/drift/*.csv`, `reports/figures/drift_*.png` |

### Comandos DVC
```bash
dvc dag          # muestra el grafo
dvc params show  # parametros rastreados desde params.yaml
dvc status       # que stages estan desactualizados
dvc repro        # reproduce el pipeline completo
dvc push         # sube artefactos al remote
```

### Nota sobre DVC push

La service account configurada no tiene cuota en Google Drive personal.
Requiere una Unidad compartida (Shared Drive) o switch a OAuth estandar.
Ver `docs/instrucciones_dvc_gdrive.md` para las dos opciones de solucion.

---

## 3. CI/CD con GitHub Actions

### Archivos

| Archivo | Descripcion |
|---|---|
| `.github/workflows/ci.yml` | Lint (ruff) + 142 tests en cada push/PR a main/dev |
| `.github/workflows/cd.yml` | Docker build + push (disparo manual hasta que exista Dockerfile) |

### CI — activo ahora

Cada push o PR a `main`/`dev` ejecuta:
1. `ruff format --check`
2. `ruff check`
3. `pytest tests/ -q` — 142 tests con datos sinteticos, sin descargar el dataset

### CD — listo pero en espera

El workflow existe y funciona. Se activa manualmente (`workflow_dispatch`)
hasta que exista el Dockerfile. Para activar el disparo automatico:
descomentar los bloques `push` y `release` en `cd.yml`.

---

## 4. Otros cambios

| Archivo | Descripcion |
|---|---|
| `.gitignore` | Patrones para credenciales GCP (key.json, *-credentials.json, etc.) |
| `docs/actualizaciones_entrega_final.md` | Cambios seccion a seccion para el reporte |
| `docs/pr_description_fastapi.md` | Este archivo |

---

## Que falta (pendiente de otros PRs)

- **Dockerfile**: el workflow `cd.yml` lo espera para activar build + push automatico
- **DVC push**: resolver el quota error de la service account (ver instrucciones)
- **Reproducibilidad en entorno limpio**: requiere Docker
- **Video de demostracion**: puede grabarse ya (CI activo, API local, Swagger); el Docker build se agrega cuando este listo

---

## Verificacion local

```bash
# Arrancar el servicio
uvicorn mibici.api.main:app --reload --port 8000

# Probar endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hora":8,"dia_semana":1,"mes":4,"es_fin_de_semana":false,
       "es_hora_pico_manana":true,"es_hora_pico_tarde":false,
       "capacidad":15.0,"lat":20.6736,"lon":-103.3435}'
curl http://localhost:8000/metrics

# Ver Swagger interactivo
start http://localhost:8000/docs

# Pipeline DVC
dvc dag
dvc status
```
