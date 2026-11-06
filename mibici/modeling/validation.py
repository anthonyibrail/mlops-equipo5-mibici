"""Esquema comun de evaluacion de modelos (Fase 2).

Todos los modelos del proyecto se evaluan igual, con lo que define params.yaml:

  - WalkForwardSplit: validacion con ventana movil. Entrena con los meses
    anteriores a un corte, evalua en el mes siguiente y avanza el corte.
    Es compatible con scikit-learn (se puede pasar como `cv` a
    RandomizedSearchCV o cross_validate).
  - SeasonalNaive y HistoricalMean: lineas base como estimadores de
    scikit-learn, para compararlas con el mismo codigo que los modelos.
  - regression_metrics y net_balance_metrics: metricas de demanda por
    estacion-hora y del balance neto (salidas - llegadas).
  - cross_validate_model: evalua un estimador en todos los cortes.

Uso (evalua las lineas base y las registra en MLflow):
    python -m mibici.modeling.validation
"""

from __future__ import annotations

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
import typer

from mibici.config import load_params
from mibici.versioning import TRACKING_URI

app = typer.Typer()


# ---------------------------------------------------------------------------
# Validacion con ventana movil
# ---------------------------------------------------------------------------
class WalkForwardSplit:
    """Particion temporal por meses con ventana de entrenamiento movil o creciente.

    Para cada corte k (k = 0 .. n_splits-1):
      prueba:        [primer_mes_prueba + k, + meses_prueba)
      entrenamiento: los `train_months` meses inmediatamente anteriores a la prueba
                     (o todo lo anterior si train_months es None).
    Nunca hay datos de prueba en entrenamiento ni datos posteriores al corte.
    """

    def __init__(
        self,
        time_col: str,
        first_test_month: str,
        n_splits: int,
        test_months: int = 1,
        train_months: int | None = None,
    ):
        self.time_col = time_col
        self.first_test_month = first_test_month
        self.n_splits = n_splits
        self.test_months = test_months
        self.train_months = train_months

    @classmethod
    def from_params(cls, params: dict) -> WalkForwardSplit:
        v = params["validacion"]
        return cls(
            time_col=v["columna_tiempo"],
            first_test_month=v["primer_mes_prueba"],
            n_splits=v["n_cortes"],
            test_months=v["meses_prueba"],
            train_months=v["meses_entrenamiento"],
        )

    def folds(self) -> pd.DataFrame:
        """Fronteras de cada corte (util para documentar y registrar en MLflow)."""
        first = pd.Timestamp(self.first_test_month + "-01")
        rows = []
        for k in range(self.n_splits):
            test_start = first + pd.DateOffset(months=k)
            test_end = test_start + pd.DateOffset(months=self.test_months)
            train_start = (
                test_start - pd.DateOffset(months=self.train_months)
                if self.train_months
                else pd.Timestamp.min
            )
            rows.append(
                {
                    "corte": k,
                    "inicio_entrenamiento": train_start,
                    "inicio_prueba": test_start,
                    "fin_prueba": test_end,
                }
            )
        return pd.DataFrame(rows)

    def split(self, X: pd.DataFrame, y=None, groups=None):
        times = pd.to_datetime(X[self.time_col]).to_numpy()
        for f in self.folds().itertuples():
            train = (times >= f.inicio_entrenamiento.to_datetime64()) & (
                times < f.inicio_prueba.to_datetime64()
            )
            test = (times >= f.inicio_prueba.to_datetime64()) & (
                times < f.fin_prueba.to_datetime64()
            )
            if not train.any() or not test.any():
                raise ValueError(f"El corte {f.corte} no tiene datos de entrenamiento o prueba")
            yield np.flatnonzero(train), np.flatnonzero(test)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Lineas base (estimadores de scikit-learn)
# ---------------------------------------------------------------------------
class SeasonalNaive(RegressorMixin, BaseEstimator):
    """Pronostico = valor de la misma estacion y hora de la semana anterior."""

    def __init__(self, lag_col: str = "salidas_lag_168h"):
        self.lag_col = lag_col

    def required_columns(self) -> list[str]:
        return [self.lag_col]

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Sin historia de una semana (solo al inicio de la serie) se pronostica 0
        return X[self.lag_col].fillna(0).to_numpy(dtype=float)


class HistoricalMean(RegressorMixin, BaseEstimator):
    """Pronostico = promedio del entrenamiento para la misma estacion, dia y hora."""

    def __init__(self, keys: tuple[str, ...] = ("estacion_id", "dia_semana", "hora")):
        self.keys = keys

    def required_columns(self) -> list[str]:
        return list(self.keys)

    def fit(self, X: pd.DataFrame, y):
        keys = list(self.keys)
        frame = X[keys].copy()
        frame["_y"] = np.asarray(y, dtype=float)
        self.table_ = frame.groupby(keys, observed=True)["_y"].mean().rename("_pred")
        self.global_mean_ = float(frame["_y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = X[list(self.keys)].join(self.table_, on=list(self.keys))["_pred"]
        # Combinaciones nunca vistas (p. ej. estacion nueva): promedio global
        return pred.fillna(self.global_mean_).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, hours, peak_hours) -> dict:
    """MAE, RMSE, MAE en horas pico y MAE relativo (MAE / demanda media)."""
    y_true = np.asarray(y_true, dtype=float)
    err = y_true - np.asarray(y_pred, dtype=float)
    peak = np.isin(np.asarray(hours), list(peak_hours))
    mae = float(np.mean(np.abs(err)))
    mean_demand = float(y_true.mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae_pico": float(np.mean(np.abs(err[peak]))) if peak.any() else float("nan"),
        "mae_relativo": mae / mean_demand if mean_demand else float("nan"),
        "demanda_media": mean_demand,
    }


def net_balance_metrics(y_out, y_in, p_out, p_in, hours, peak_hours) -> dict:
    """Metricas del balance neto (salidas - llegadas) a partir de ambos pronosticos."""
    m = regression_metrics(
        np.asarray(y_out, float) - np.asarray(y_in, float),
        np.asarray(p_out, float) - np.asarray(p_in, float),
        hours,
        peak_hours,
    )
    m.pop("mae_relativo")  # el balance neto puede promediar cerca de 0
    return {f"balance_{k}": v for k, v in m.items()}


# ---------------------------------------------------------------------------
# Evaluacion en todos los cortes
# ---------------------------------------------------------------------------
def cross_validate_model(
    estimator,
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    splitter: WalkForwardSplit,
    peak_hours,
    return_predictions: bool = False,
):
    """Entrena y evalua `estimator` en cada corte. Devuelve una fila de metricas por corte.

    Si return_predictions=True devuelve ademas una Serie con el pronostico de prueba
    de cada fila (indice de df), util para calcular el balance neto.
    """
    rows, preds = [], []
    folds = splitter.folds()
    for k, (train_idx, test_idx) in enumerate(splitter.split(df)):
        model = clone(estimator)
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        model.fit(train[feature_cols], train[target])
        pred = model.predict(test[feature_cols])
        metrics = regression_metrics(test[target], pred, test["hora"], peak_hours)
        rows.append(
            {
                "corte": k,
                "mes_prueba": folds.loc[k, "inicio_prueba"].strftime("%Y-%m"),
                "filas_entrenamiento": len(train),
                "filas_prueba": len(test),
                **metrics,
            }
        )
        if return_predictions:
            preds.append(pd.Series(pred, index=test.index))
    result = pd.DataFrame(rows)
    if return_predictions:
        return result, pd.concat(preds)
    return result


def summarize(fold_metrics: pd.DataFrame) -> dict:
    """Promedio y desviacion estandar de cada metrica entre cortes."""
    cols = [c for c in fold_metrics.columns if c.startswith(("mae", "rmse", "balance"))]
    out = {}
    for c in cols:
        out[f"{c}_promedio"] = float(fold_metrics[c].mean())
        out[f"{c}_desv"] = float(fold_metrics[c].std(ddof=0))
    return out


def baseline_estimators(params: dict, target: str) -> dict[str, BaseEstimator]:
    lb = params["lineas_base"]
    return {
        "ingenuo_estacional": SeasonalNaive(lb["columna_estacional"].format(objetivo=target)),
        "promedio_historico": HistoricalMean(tuple(lb["llaves_promedio"])),
    }


# ---------------------------------------------------------------------------
# CLI: lineas base con validacion movil, registradas en MLflow
# ---------------------------------------------------------------------------
@app.command()
def main():
    params = load_params()
    splitter = WalkForwardSplit.from_params(params)
    peak = params["problema"]["horas_pico"]
    targets = params["problema"]["objetivos"]

    df = pd.read_parquet(params["datos"]["features"])
    logger.info(f"Panel: {len(df):,} filas; cortes:\n{splitter.folds()}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])

    for name in ["ingenuo_estacional", "promedio_historico"]:
        folds_by_target, preds = {}, {}
        for target in targets:
            est = baseline_estimators(params, target)[name]
            folds_by_target[target], preds[target] = cross_validate_model(
                est,
                df,
                target,
                est.required_columns(),
                splitter,
                peak,
                return_predictions=True,
            )
        idx = preds[targets[0]].index
        test = df.loc[idx]
        balance = (
            net_balance_metrics(
                test["salidas"],
                test["llegadas"],
                preds["salidas"],
                preds["llegadas"],
                test["hora"],
                peak,
            )
            if set(targets) >= {"salidas", "llegadas"}
            else {}
        )

        with mlflow.start_run(run_name=f"linea_base_{name}"):
            mlflow.set_tags({"tipo": "linea_base", "modelo": name, "dataset_version": "v3"})
            mlflow.log_params(
                {
                    "modelo": name,
                    "objetivos": ",".join(targets),
                    "n_cortes": splitter.n_splits,
                    "primer_mes_prueba": splitter.first_test_month,
                    "meses_entrenamiento": splitter.train_months,
                }
            )
            for target, folds in folds_by_target.items():
                for row in folds.itertuples():
                    mlflow.log_metric(f"{target}_mae_corte", row.mae, step=row.corte)
                mlflow.log_metrics({f"{target}_{k}": v for k, v in summarize(folds).items()})
                mlflow.log_table(folds, f"cortes_{target}.json")
            mlflow.log_metrics(balance)
        logger.success(
            f"{name}: "
            + ", ".join(
                f"{t} MAE={summarize(f)['mae_promedio']:.3f}" for t, f in folds_by_target.items()
            )
            + (f", balance MAE={balance['balance_mae']:.3f}" if balance else "")
        )


if __name__ == "__main__":
    app()
