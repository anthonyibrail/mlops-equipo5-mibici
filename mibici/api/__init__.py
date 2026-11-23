"""Servicio REST de pronostico de demanda MiBici (Entrega Final).

Estructura del modulo:
    schemas.py  -- Modelos Pydantic de entrada y salida (COMPLETO).
    loader.py   -- Carga de pipelines pkl desde models/ (PENDIENTE).
    main.py     -- App FastAPI con /health, /predict y /metrics (PENDIENTE).

Para arrancar el servicio (una vez implementado loader.py y main.py):
    uvicorn mibici.api.main:app --reload --port 8000

Para ver la documentacion interactiva:
    http://localhost:8000/docs
"""
