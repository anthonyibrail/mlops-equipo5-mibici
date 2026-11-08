"""Pruebas de la promocion a @champion y de la carga del modelo desde el Model Registry.

Usan un MLflow temporal (SQLite en tmp_path) con modelos diminutos, por lo que no
requieren el dataset ni tocan el mlflow.db del proyecto.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mibici.modeling import predict, promote

CRITERIA = ["salidas", "llegadas", "balance"]


def _params():
    return {
        "problema": {"objetivos": ["salidas", "llegadas"]},
        "mlflow": {"experimento_modelos": "prueba-modelos", "modelo_registrado": "prueba-demanda"},
        "promocion": {
            "alias_candidato": "candidate",
            "alias_produccion": "champion",
            "linea_base_referencia": "promedio_historico",
            "criterios": CRITERIA,
        },
    }


def _report(mae):
    """Reporte de seis cortes con el mismo MAE en cada corte."""
    return pd.DataFrame(
        [{"corte": k, "objetivo": c, "mae": m} for c, m in mae.items() for k in range(6)]
    )


# --- Funciones puras -----------------------------------------------------------
def test_model_scores_promedia_los_cortes():
    report = pd.DataFrame({"objetivo": ["salidas"] * 2, "mae": [0.6, 0.8]})
    assert promote.model_scores(report, ["salidas"]) == {"salidas": pytest.approx(0.7)}
    with pytest.raises(ValueError):
        promote.model_scores(report, ["llegadas"])


def test_nombres_de_metricas_de_linea_base():
    assert promote.baseline_metric_name("salidas") == "salidas_mae_promedio"
    assert promote.baseline_metric_name("balance") == "balance_mae"


def test_promueve_solo_si_supera_en_todos_los_criterios():
    baseline = {"salidas": 0.70, "llegadas": 0.70, "balance": 0.75}
    ok, table = promote.decide_promotion({"salidas": 0.67, "llegadas": 0.66, "balance": 0.72}, baseline)
    assert ok
    assert table.loc[table["criterio"] == "salidas", "mejora_pct"].item() == pytest.approx(100 * 0.03 / 0.70)

    rejected, table = promote.decide_promotion({"salidas": 0.67, "llegadas": 0.66, "balance": 0.80}, baseline)
    assert not rejected
    assert table.loc[~table["supera"], "criterio"].tolist() == ["balance"]


# --- Flujo completo con un MLflow temporal ---------------------------------------
@pytest.fixture
def registry(tmp_path, monkeypatch):
    """MLflow temporal con una linea base y dos modelos registrados como @candidate."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    params = _params()
    monkeypatch.setattr(promote, "TRACKING_URI", uri)
    monkeypatch.setattr(predict, "TRACKING_URI", uri)
    monkeypatch.setattr(promote, "load_params", lambda: params)
    monkeypatch.setattr(promote, "REPORT_PATH", tmp_path / "walk_forward_6_cortes.csv")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()

    mlflow.set_experiment(params["mlflow"]["experimento_modelos"])
    with mlflow.start_run(run_name="linea_base_promedio_historico"):
        mlflow.set_tags({"tipo": "linea_base", "modelo": "promedio_historico"})
        mlflow.log_metrics({"salidas_mae_promedio": 0.70, "llegadas_mae_promedio": 0.70, "balance_mae": 0.75})

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    for value, target in ((1.5, "salidas"), (2.5, "llegadas")):
        name = f"prueba-demanda-{target}"
        with mlflow.start_run():
            model = DummyRegressor(strategy="constant", constant=value).fit(X, [value] * 3)
            mlflow.sklearn.log_model(sk_model=model, name="modelo", registered_model_name=name)
        version = client.search_model_versions(f"name = '{name}'")[0].version
        client.set_registered_model_alias(name, "candidate", version)
    return client, params, tmp_path


def test_promocion_asigna_champion_y_predict_lo_carga(registry):
    client, params, tmp_path = registry
    _report({"salidas": 0.67, "llegadas": 0.66, "balance": 0.72}).to_csv(promote.REPORT_PATH, index=False)

    promote.main()

    for target, value in (("salidas", 1.5), ("llegadas", 2.5)):
        name = f"prueba-demanda-{target}"
        champion = client.get_model_version_by_alias(name, "champion")
        assert champion.tags["promocion"] == "champion"
        assert float(champion.tags["mejora_pct_balance"]) == pytest.approx(4.0)
        model = predict.load_champion(params, target)
        assert np.allclose(model.predict(pd.DataFrame({"x": [5.0]})), value)


def test_promocion_rechazada_no_asigna_champion(registry):
    client, params, tmp_path = registry
    _report({"salidas": 0.67, "llegadas": 0.66, "balance": 0.80}).to_csv(promote.REPORT_PATH, index=False)

    promote.main()

    for target in ("salidas", "llegadas"):
        name = f"prueba-demanda-{target}"
        assert "champion" not in client.get_registered_model(name).aliases
        assert client.get_model_version_by_alias(name, "candidate").tags["promocion"] == "rechazada"
    with pytest.raises(RuntimeError):
        predict.load_champion(params, "salidas")