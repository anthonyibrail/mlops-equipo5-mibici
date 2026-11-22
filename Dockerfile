# Servicio FastAPI de pronostico de demanda MiBici.
#
# Requisito previo (fuera de Docker): make export-model, que guarda en
# models/champion/ los modelos @champion del Model Registry y model_info.json.
#
# Construir y correr:
#   docker build -t ml-service:latest .
#   docker run -p 8000:8000 ml-service:latest
# Swagger: http://localhost:8000/docs

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    # Un hilo por prediccion: con todos los nucleos, servir una fila tardaba
    # ~3.8 s por la coordinacion de hilos de OpenMP; con uno, ~0.012 s.
    OMP_NUM_THREADS=1

WORKDIR /app

# 1) Dependencias primero: esta capa se reutiliza mientras no cambien.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) Codigo y configuracion del proyecto.
COPY params.yaml ./
COPY mibici/ mibici/

# 3) Modelos @champion exportados y metricas de validacion para /metrics.
COPY models/champion/ models/
COPY reports/model_selection/ reports/model_selection/

# 4) El servicio no corre como root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
