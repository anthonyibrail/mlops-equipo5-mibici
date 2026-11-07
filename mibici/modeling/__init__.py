"""Modulo de modelado: entrenamiento, evaluacion e inferencia."""

import numpy as np


def bool_to_int(X: np.ndarray) -> np.ndarray:
    """Convierte un array booleano a entero (0/1).

    Se usa como transformacion en el ColumnTransformer del Pipeline para que
    los flags booleanos (es_fin_de_semana, etc.) sean tratados como numericos
    por HistGradientBoostingRegressor. Definida aqui (fuera de train.py o
    predict.py) para que el pickle del Pipeline resuelva el modulo
    correctamente en cualquier contexto de carga.
    """
    return X.astype(int)
