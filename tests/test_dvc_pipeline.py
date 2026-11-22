"""Validacion de la definicion del pipeline DVC (dvc.yaml).

Estos tests verifican que dvc.yaml sea sintacticamente correcto y coherente
con la estructura del proyecto. No ejecutan el pipeline ni requieren datos.

Ejecutar con: pytest tests/test_dvc_pipeline.py -v
"""

import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
DVC_YAML_PATH = ROOT / "dvc.yaml"

EXPECTED_STAGES = {"data", "features", "train", "predict"}

# Archivos Python que deben existir como deps en cada stage
EXPECTED_DEPS = {
    "data": [
        "mibici/dataset.py",
        "mibici/cleaning.py",
        "mibici/versioning.py",
    ],
    "features": [
        "mibici/features.py",
        "mibici/versioning.py",
    ],
    "train": [
        "mibici/modeling/tune.py",
        "mibici/modeling/validation.py",
        "mibici/modeling/register.py",
        "mibici/modeling/promote.py",
        "mibici/modeling/__init__.py",
    ],
    "predict": [
        "mibici/modeling/predict.py",
    ],
}


def _load_dvc_yaml():
    """Carga y retorna el contenido de dvc.yaml."""
    with open(DVC_YAML_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Tests de existencia y estructura basica
# ---------------------------------------------------------------------------


def test_dvc_yaml_exists():
    """dvc.yaml debe existir en la raiz del proyecto."""
    assert DVC_YAML_PATH.exists(), (
        f"dvc.yaml no encontrado en {ROOT}. "
        "Ejecuta 'dvc init' y crea el archivo segun la documentacion."
    )


def test_dvc_yaml_is_valid_yaml():
    """dvc.yaml debe ser YAML valido (parseable sin errores)."""
    content = _load_dvc_yaml()
    assert content is not None, "dvc.yaml esta vacio o no es YAML valido."


def test_dvc_yaml_has_stages_section():
    """dvc.yaml debe tener una seccion 'stages'."""
    content = _load_dvc_yaml()
    assert "stages" in content, (
        "dvc.yaml debe tener una seccion 'stages'. "
        "Consulta la documentacion de DVC en docs/instrucciones_dvc_gdrive.md."
    )


# ---------------------------------------------------------------------------
# Tests de stages esperados
# ---------------------------------------------------------------------------


def test_all_expected_stages_present():
    """El pipeline debe contener exactamente los stages data, features, train y predict."""
    content = _load_dvc_yaml()
    stages = set(content["stages"].keys())
    missing = EXPECTED_STAGES - stages
    assert not missing, (
        f"Stages faltantes en dvc.yaml: {missing}. "
        f"Stages actuales: {stages}"
    )


def test_each_stage_has_cmd():
    """Cada stage debe tener un campo 'cmd' (el comando que ejecuta)."""
    content = _load_dvc_yaml()
    for stage_name, stage_def in content["stages"].items():
        assert "cmd" in stage_def, (
            f"Stage '{stage_name}' no tiene campo 'cmd'. "
            "Revisa dvc.yaml."
        )
        assert stage_def["cmd"], f"Stage '{stage_name}': 'cmd' esta vacio."


def test_each_stage_has_deps():
    """Cada stage debe declarar al menos una dependencia."""
    content = _load_dvc_yaml()
    for stage_name, stage_def in content["stages"].items():
        assert "deps" in stage_def, (
            f"Stage '{stage_name}' no tiene campo 'deps'. "
            "Las dependencias permiten a DVC detectar cambios."
        )
        assert len(stage_def["deps"]) > 0, (
            f"Stage '{stage_name}': lista 'deps' esta vacia."
        )


# ---------------------------------------------------------------------------
# Tests de coherencia con el codigo fuente
# ---------------------------------------------------------------------------


def test_python_deps_exist_in_codebase():
    """Los archivos .py listados como deps deben existir en el repositorio."""
    content = _load_dvc_yaml()
    for stage_name, stage_def in content["stages"].items():
        for dep in stage_def.get("deps", []):
            if str(dep).endswith(".py"):
                dep_path = ROOT / dep
                assert dep_path.exists(), (
                    f"Stage '{stage_name}': dependencia '{dep}' no existe "
                    f"en el repositorio ({dep_path})."
                )


def test_params_yaml_referenced_as_dep():
    """params.yaml debe ser referenciado por al menos un stage."""
    content = _load_dvc_yaml()
    stages_with_params = [
        name
        for name, defn in content["stages"].items()
        if "params" in defn
    ]
    assert len(stages_with_params) > 0, (
        "Ningun stage referencia params.yaml. "
        "Al menos 'train' y 'features' deberian tener 'params'."
    )


def test_tuning_winners_is_dep_of_train():
    """tuning_winners.json debe ser dep de 'train' para evitar re-busqueda."""
    content = _load_dvc_yaml()
    train_deps = content["stages"]["train"].get("deps", [])
    winners_file = "reports/model_selection/tuning_winners.json"
    assert winners_file in train_deps, (
        f"'{winners_file}' no esta en los deps de la etapa 'train'. "
        "Sin esto, DVC reejecutaria la busqueda de hiperparametros "
        "cada vez que cambie cualquier otro dep."
    )


def test_walk_forward_csv_is_metric():
    """walk_forward_6_cortes.csv debe estar declarado como metrica DVC."""
    content = _load_dvc_yaml()
    train_metrics = content["stages"]["train"].get("metrics", [])
    # Las metricas pueden ser lista de strings o lista de dicts
    metric_paths = []
    for item in train_metrics:
        if isinstance(item, str):
            metric_paths.append(item)
        elif isinstance(item, dict):
            metric_paths.extend(item.keys())
    csv_path = "reports/model_selection/walk_forward_6_cortes.csv"
    assert csv_path in metric_paths, (
        f"'{csv_path}' no esta declarado como 'metrics' en la etapa 'train'. "
        "Declararlo permite usar 'dvc metrics show' para ver el MAE del modelo."
    )


# ---------------------------------------------------------------------------
# Tests de outputs rastreados
# ---------------------------------------------------------------------------


def test_data_stage_produces_interim_parquets():
    """La etapa 'data' debe producir los parquets intermedios."""
    content = _load_dvc_yaml()
    data_outs = content["stages"]["data"].get("outs", [])
    expected = {
        "data/interim/trips_v0.parquet",
        "data/interim/trips_v1.parquet",
    }
    assert expected.issubset(set(data_outs)), (
        f"Outputs faltantes en etapa 'data': {expected - set(data_outs)}"
    )


def test_features_stage_produces_v3_panel():
    """La etapa 'features' debe producir el panel v3."""
    content = _load_dvc_yaml()
    features_outs = content["stages"]["features"].get("outs", [])
    assert "data/processed/demand_features_v3.parquet" in features_outs, (
        "El panel v3 no esta en los outputs de la etapa 'features'."
    )


def test_train_stage_produces_pkl_models():
    """La etapa 'train' debe producir los pipelines pkl."""
    content = _load_dvc_yaml()
    train_outs = content["stages"]["train"].get("outs", [])
    expected = {
        "models/pipeline_salidas.pkl",
        "models/pipeline_llegadas.pkl",
    }
    assert expected.issubset(set(train_outs)), (
        f"Outputs faltantes en etapa 'train': {expected - set(train_outs)}"
    )


def test_predict_stage_produces_predicciones():
    """La etapa 'predict' debe producir el parquet de predicciones."""
    content = _load_dvc_yaml()
    predict_outs = content["stages"]["predict"].get("outs", [])
    assert "data/processed/predicciones.parquet" in predict_outs, (
        "predicciones.parquet no esta en los outputs de la etapa 'predict'."
    )


# ---------------------------------------------------------------------------
# Tests de orden del pipeline (dependencias entre stages)
# ---------------------------------------------------------------------------


def test_features_depends_on_data_output():
    """'features' debe depender del output de 'data' para que el orden sea correcto."""
    content = _load_dvc_yaml()
    features_deps = content["stages"]["features"].get("deps", [])
    assert "data/interim/trips_v1.parquet" in features_deps, (
        "'features' no depende de trips_v1.parquet (output de 'data'). "
        "Sin esto, DVC no puede deducir el orden del pipeline."
    )


def test_train_depends_on_features_output():
    """'train' debe depender del output de 'features'."""
    content = _load_dvc_yaml()
    train_deps = content["stages"]["train"].get("deps", [])
    assert "data/processed/demand_features_v3.parquet" in train_deps, (
        "'train' no depende del panel v3 (output de 'features')."
    )


def test_predict_depends_on_train_outputs():
    """'predict' debe depender de ambos pkl generados por 'train'."""
    content = _load_dvc_yaml()
    predict_deps = content["stages"]["predict"].get("deps", [])
    expected = {
        "models/pipeline_salidas.pkl",
        "models/pipeline_llegadas.pkl",
    }
    assert expected.issubset(set(predict_deps)), (
        f"'predict' no depende de todos los modelos de 'train': "
        f"{expected - set(predict_deps)}"
    )
