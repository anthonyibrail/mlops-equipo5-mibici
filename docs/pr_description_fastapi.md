# PR: feat(api) — Implementacion completa del servicio FastAPI MiBici

**Branch:** `feature/fastapi` → `main`

---

## Que hace este PR

Implementa el servicio REST de pronostico de demanda MiBici (Entrega Final),
completando los tres endpoints especificados en el reporte de Fase 3.

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `mibici/api/loader.py` | Implementa `load_pipelines()` y `get_pipelines()` |
| `mibici/api/main.py` | Implementa lifespan + endpoints /health, /predict, /metrics |
| `tests/test_api.py` | 15 tests activos (todos pasan) |

### Contenido previo (ya en la branch desde PR anterior)

| Archivo | Descripcion |
|---|---|
| `mibici/api/schemas.py` | Schemas Pydantic completos (contrato de la API) |
| `mibici/api/__init__.py` | Marca el paquete |
| `dvc.yaml` | Pipeline DVC de 5 stages |
| `.github/workflows/ci.yml` | Lint + tests en cada push/PR |
| `.github/workflows/cd.yml` | Docker build (manual hasta Dockerfile) |
| `docs/instrucciones_dvc_gdrive.md` | Instrucciones Google Drive + troubleshooting |

---

## Endpoints implementados

### GET /health
```json
// Response 200
{"status": "ok", "source": "local", "targets": ["salidas", "llegadas"]}
```

### POST /predict
```json
// Request
{
  "hora": 8, "dia_semana": 1, "mes": 4,
  "es_fin_de_semana": false, "es_hora_pico_manana": true, "es_hora_pico_tarde": false,
  "capacidad": 15.0, "lat": 20.6736, "lon": -103.3435
  // lags opcionales — None se convierte a NaN (HistGBR lo maneja nativamente)
}

// Response 200
{"salidas_pred": 4.1, "llegadas_pred": 2.8, "balance": 1.3}
```

### GET /metrics
```json
// Response 200
{
  "mae_salidas": 0.674, "mae_llegadas": 0.662, "mae_balance": 0.716,
  "n_folds": 6,
  "source": "reports/model_selection/walk_forward_6_cortes.csv"
}
```

---

## Tests

```
pytest tests/test_api.py -v
# 15 passed in 6.21s

pytest tests/ -q
# 142 passed, 4 warnings in 48.46s
```

Todos los tests usan pipelines mockeados; no requieren los pkl reales.

---

## Decisiones de implementacion

**Lifespan FastAPI**: se usa `@asynccontextmanager` (FastAPI >= 0.95) para
cargar los modelos al arrancar el servicio y almacenarlos en `app.state.pipelines`.
Esto evita I/O en cada request.

**bool_to_int y pickle**: la funcion `bool_to_int` esta en `mibici/modeling/__init__.py`
(no en `train.py`) para que el pickle del Pipeline sea estable al cargarse desde
cualquier modulo. `loader.py` importa `bool_to_int` con `# noqa: F401` para garantizar
que el nombre quede registrado en el namespace antes de `pickle.load`.

**Ruta de metricas**: usa `REPORTS_DIR` de `mibici.config` (path absoluto basado en
el proyecto), no un path relativo, para que funcione independientemente del
directorio de trabajo.

**Tests mockeados**: se parchea `mibici.api.loader.load_pipelines` en el atributo del
modulo (no la importacion local) para que el lifespan use el mock al arrancar el TestClient.

---

## Que falta (pendiente de otros PRs)

- **Dockerfile + CD**: el workflow `cd.yml` esta listo pero inactivo hasta que exista el Dockerfile
- **DVC push a Google Drive**: la service account no tiene cuota en Drive personal; requiere Shared Drive o switch a OAuth estandar (ver `docs/instrucciones_dvc_gdrive.md`)
- **Reproducibilidad en entorno limpio**: requiere Docker
- **Video de demostracion**: requiere que el servicio corra en Docker

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

# Ver Swagger
start http://localhost:8000/docs
```
