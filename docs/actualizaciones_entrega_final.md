# Actualizaciones pendientes en Entrega_Final_Equipo05.docx

Este archivo detalla, seccion a seccion, los cambios que el equipo debe hacer
en el documento de entrega final para reflejar lo implementado hasta hoy.

---

## Encabezado — Imagen en DockerHub

**Cambio:** Reemplazar el placeholder por la URL real cuando se suba la imagen.

```
Imagen en DockerHub: [PENDIENTE: liga a la imagen y tag versionado]
```

> Sigue pendiente. Requiere el Dockerfile (ver seccion 2.3.4).

---

## Resumen ejecutivo — Tabla de herramientas (seccion 1.3)

**Fila "Calidad"** — agregar GitHub Actions:

```
Calidad | pytest, ruff, GitHub con rama protegida y Pull Requests,
         GitHub Actions CI (lint + tests en cada PR a main/dev)
```

**Fila "Servicio y portabilidad"** — FastAPI esta completo:

```
Servicio y portabilidad | FastAPI (implementado), Pydantic (implementado),
                          Docker (pendiente), DockerHub (pendiente)
```

---

## Tabla de roles — Columna "Entrega final"

Reemplazar los tres [PENDIENTE] con:

| Rol | Entrega final |
|---|---|
| Data Scientist | [Descripcion de su contribucion - confirmar con el integrante] |
| Feature / ML Engineer | Servicio FastAPI completo: loader.py (carga de pkl), endpoints /health, /predict y /metrics con validacion Pydantic; 15 tests de integracion de la API con pipelines mockeados |
| MLOps / Software Engineer | Pipeline DVC reproducible (5 stages: data, features, train, predict, drift); workflows GitHub Actions CI (lint + 142 tests) y CD (Docker, manual hasta Dockerfile); configuracion de remote Google Drive con service account |

---

## Seccion 2.3.1 — Pruebas unitarias y de integración

**Reemplazar el [PENDIENTE] con:**

```
Se agregaron pruebas de integracion de la API REST con FastAPI TestClient
y pipelines mockeados. Las pruebas cubren:
- GET /health: estado del servicio y objetivos disponibles
- POST /predict: predicciones con payload valido, lags nulos, 
  validacion de campos invalidos (hora=25 → 422, campo faltante → 422)
- GET /metrics: valores MAE finitos y positivos, n_folds=6

Numero final de pruebas: 142 (103 originales + 24 DVC pipeline + 15 API).
Todas usan datos sinteticos o mocks; no requieren descargar el dataset.

Captura de ejecucion:
  pytest tests/ -q → 142 passed, 4 warnings in 48.46s
```

---

## Seccion 2.3.2 — Servicio del modelo con FastAPI

**Reemplazar el [PENDIENTE] completo con:**

```
El servicio esta implementado en mibici/api/ con tres modulos:

- schemas.py: modelos Pydantic de entrada (16 features del panel v3,
  lags opcionales → NaN nativo de HistGBR) y salida.
- loader.py: carga los pkl de models/ al arrancar la app. Usa bool_to_int
  desde mibici.modeling para garantizar la estabilidad del pickle.
- main.py: app FastAPI con lifespan que carga los modelos en app.state.

Endpoints:
  GET  /health  → {"status": "ok", "source": "local", "targets": ["salidas","llegadas"]}
  POST /predict → salidas_pred, llegadas_pred, balance (siempre >= 0 por np.clip)
  GET  /metrics → MAE salidas/llegadas/balance de walk_forward_6_cortes.csv

Ruta del artefacto: models/pipeline_salidas.pkl y models/pipeline_llegadas.pkl
(generados por dvc repro train, que incluye register.py que los sube al Registry
como models:/mibici-demanda-salidas@champion y models:/mibici-demanda-llegadas@champion).

Ejemplo de peticion (curl):
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"hora":8,"dia_semana":1,"mes":4,"es_fin_de_semana":false,
         "es_hora_pico_manana":true,"es_hora_pico_tarde":false,
         "capacidad":15.0,"lat":20.6736,"lon":-103.3435}'

Ejemplo de respuesta:
  {"salidas_pred": 4.1, "llegadas_pred": 2.8, "balance": 1.3}

Documentacion interactiva disponible en http://localhost:8000/docs
(Swagger UI generado automaticamente por FastAPI).

Para arrancar:
  uvicorn mibici.api.main:app --reload --port 8000
```

---

## Seccion 2.3.3 — Reproducibilidad en otro entorno

> Sigue pendiente. Requiere el Dockerfile para probar en entorno limpio.
> La base existe: requirements.txt con versiones fijadas, Python 3.11,
> semilla 42, dvc.lock con hashes de todos los artefactos.

---

## Seccion 2.3.4 — Contenedor Docker

> Sigue pendiente. El workflow de CD (.github/workflows/cd.yml) esta
> preparado y se activa manualmente; solo necesita el Dockerfile.
> Ver README seccion 6 para los comandos previstos.

---

## Seccion 2.3.5 — Simulacion de data drift

**Reemplazar la nota [PENDIENTE] en la tabla:**

```
[Resultados confirmados con dvc repro drift usando el modelo @champion.
Ver reports/drift/resumen_drift.csv y reports/drift/psi_por_columna.csv]
```

Los resultados del `dvc repro` completo confirman la tabla existente.
Las imagenes (Figura 3 y Figura 4) se generaron en `reports/figures/`.

---

## Seccion 4.1 — Lecciones aprendidas

**Reemplazar el [PENDIENTE] con:**

```
- Separar el contrato de la API (schemas Pydantic) de la logica de carga
  del modelo (loader.py) facilita el testing sin necesidad de pkl reales:
  basta con mockear load_pipelines() antes de que el lifespan arranque.
- DVC complementa MLflow: MLflow rastrea experimentos y el Model Registry,
  DVC rastrea los artefactos grandes (parquets, pkl) y la reproducibilidad
  del pipeline. Ambos son necesarios para un flujo MLOps completo.
- Las Service Accounts de Google Cloud no tienen cuota en Google Drive
  personal; para usar DVC con service account se necesita un Shared Drive
  (Google Workspace) o volver a OAuth estandar.
- La dependencia entre validation.py (lineas base en MLflow) y promote.py
  (comparacion contra esas lineas base) debe reflejarse en el orden del
  pipeline DVC; omitirla provoca ValueError en el stage train.
```

---

## Seccion 4.3 — Trabajo futuro y recomendaciones

**Eliminar el item de GitHub Actions** (ya esta implementado) y reemplazar con:

```
- Integrar el Dockerfile al workflow de CD (.github/workflows/cd.yml)
  para construccion y publicacion automatica de la imagen Docker.
- Ejecutar el monitoreo de drift cada mes con los datos reales que
  publica MiBici y automatizar el reentrenamiento cuando haya alerta.
- Convertir el balance pronosticado en una recomendacion de rutas
  para los camiones (decision final del ML Canvas).
```

---

## Declaracion de uso de IA

**Agregar al final:**

```
[Agregar otras herramientas de IA usadas por el equipo, si las hay.
Ejemplo: GitHub Copilot, ChatGPT, Gemini, etc.]
```

---

## Items que siguen pendientes (no hay avances todavia)

| Item | Seccion | Descripcion |
|---|---|---|
| Dockerfile | 2.3.4 | Construccion y publicacion de imagen Docker |
| Reproducibilidad | 2.3.3 | Prueba en entorno limpio con Docker |
| DockerHub link | Encabezado | URL de la imagen publicada |
| DVC push a GDrive | - | Requiere Shared Drive o switch a OAuth |
| Video demostracion | 4 Entregas | Video de 3-5 minutos |
