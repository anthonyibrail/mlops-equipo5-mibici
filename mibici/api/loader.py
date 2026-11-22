"""Carga de modelos para el servicio MiBici.

PENDIENTE (Feature / ML Engineer):
    Implementar load_pipelines() para que cargue los pipelines pkl
    desde models/ al arrancar el servicio.

Notas de implementacion:
    1. Importar pickle y abrir MODELS_DIR / "pipeline_{target}.pkl".
    2. bool_to_int DEBE importarse desde mibici.modeling (no desde __main__)
       para que pickle.load funcione correctamente desde este modulo.
       Ya esta en mibici/modeling/__init__.py -- no hay que mover nada.
    3. Los pipelines son objetos sklearn.pipeline.Pipeline con un
       ColumnTransformer + HistGradientBoostingRegressor.
    4. Cargar al arrancar (no por request) y devolver un singleton
       para evitar I/O en cada peticion.

Uso esperado en main.py:
    from mibici.api.loader import get_pipelines
    pipelines = get_pipelines()   # dict: {"salidas": Pipeline, "llegadas": Pipeline}
    pred = pipelines["salidas"].predict(X)

Referencia:
    - mibici/modeling/predict.py  -> load_pipeline() y load_champion()
    - mibici/config.py            -> MODELS_DIR
    - mibici/modeling/__init__.py -> bool_to_int (pickle-stable)
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# TODO: implementar las funciones de abajo.
#       Referencia: mibici/modeling/predict.py
# ---------------------------------------------------------------------------

_pipelines: dict[str, Pipeline] | None = None


def load_pipelines() -> dict[str, Pipeline]:
    """Carga y retorna los pipelines pkl desde models/.

    TODO:
        import pickle
        from mibici.config import MODELS_DIR
        from mibici.modeling import bool_to_int  # noqa: F401 - necesario para pickle

        pipelines = {}
        for target in ("salidas", "llegadas"):
            path = MODELS_DIR / f"pipeline_{target}.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Modelo no encontrado: {path}. "
                    "Ejecuta 'dvc repro train' o 'make train' primero."
                )
            with open(path, "rb") as fh:
                pipelines[target] = pickle.load(fh)
        return pipelines

    Returns:
        dict con claves "salidas" y "llegadas", cada una con su Pipeline.

    Raises:
        FileNotFoundError: si algun pkl no existe en models/.
    """
    raise NotImplementedError(
        "load_pipelines() no esta implementado. "
        "Ver mibici/api/loader.py para las instrucciones."
    )


def get_pipelines() -> dict[str, Pipeline]:
    """Retorna el singleton de pipelines, cargandolo si no existe.

    TODO: descomentar el bloque de abajo una vez que load_pipelines() funcione.

        global _pipelines
        if _pipelines is None:
            _pipelines = load_pipelines()
        return _pipelines
    """
    raise NotImplementedError(
        "get_pipelines() no esta implementado. "
        "Implementa load_pipelines() primero."
    )
