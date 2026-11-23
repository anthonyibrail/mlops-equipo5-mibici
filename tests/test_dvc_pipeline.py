"""Validacion de la definicion del pipeline DVC (dvc.yaml).

Estos tests verifican que dvc.yaml sea sintacticamente correcto y coherente
con la estructura del proyecto. No ejecutan el pipeline ni requieren datos.

Ejecutar con: pytest tests/test_dvc_pipeline.py -v
"""

import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
DVC_YAML_PATH = ROOT / "dvc.yaml"

EXPECTED_STAGES = {"data", "features", "train", "predict", "drift"}

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
        "mibici/modeling/validation.py",
        "mibici/modeling/train.py",
        "mibici/modeling/tune.py",
        "mibici/modeling/register.py",
        "mibici/modeling/promote.py",
        "mibici/modeling/__init__.py",
    ],
    "predict": [
        "mibici/modeling/predict.py",
    ],
    "drift": [
        "mibici/modeling/drift.py",
        "mibici/modeling/predict.py",
        "mibici/modeling/__init__.py",
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


def test_train_cmd_includes_train_py():
    """El stage 'train' debe ejecutar python -m mibici.modeling.train para crear los pkl.

    train.py es quien guarda models/pipeline_salidas.pkl y models/pipeline_llegadas.pkl.
    register.py solo sube al MLflow Model Registry, no sobreescribe los pkl locales.
    predict --source local necesita esos pkl, por lo que train.py es obligatorio.
    """
    content = _load_dvc_yaml()
    cmd = content["stages"]["train"]["cmd"]
    assert "mibici.modeling.train" in cmd, (
        "El stage 'train' debe incluir 'python -m mibici.modeling.train'. "
        "Sin esto, los pkl en models/ no se crean y predict --source local falla."
    )


def test_train_cmd_includes_validation_before_promote():
    """El comando de 'train' debe ejecutar validation antes de promote.

    promote.py busca runs de la linea base 'promedio_historico' en MLflow;
    si validation no corrio primero, promote falla con ValueError.
    """
    content = _load_dvc_yaml()
    cmd = content["stages"]["train"]["cmd"]
    assert "mibici.modeling.validation" in cmd, (
        "El stage 'train' debe ejecutar 'python -m mibici.modeling.validation' "
        "ANTES de promote. Sin esto, promote.py no encuentra la linea base "
        "en MLflow y lanza ValueError."
    )
    assert "mibici.modeling.promote" in cmd, (
        "El stage 'train' debe incluir 'python -m mibici.modeling.promote'."
    )
    # Verifica que validation aparece antes que promote en el comando
    idx_validation = cmd.find("mibici.modeling.validation")
    idx_promote = cmd.find("mibici.modeling.promote")
    assert idx_validation < idx_promote, (
        "En el stage 'train', 'validation' debe aparecer ANTES que 'promote' "
        "en el campo 'cmd'."
    )
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


# ---------------------------------------------------------------------------
# Tests del stage drift
# ---------------------------------------------------------------------------


def test_drift_stage_exists():
    """El pipeline debe tener un stage 'drift' para la simulacion de data drift."""
    content = _load_dvc_yaml()
    assert "drift" in content["stages"], (
        "No existe el stage 'drift' en dvc.yaml. "
        "El Makefile pipeline-train incluye 'make drift' como ultima etapa."
    )


def test_drift_uses_local_source():
    """El stage 'drift' debe usar --source local para leer los pkl del DVC pipeline."""
    content = _load_dvc_yaml()
    cmd = content["stages"]["drift"].get("cmd", "")
    assert "--source local" in cmd, (
        "El stage 'drift' debe usar '--source local' para cargar los pkl "
        "gestionados por DVC en lugar del MLflow Model Registry."
    )


def test_drift_depends_on_train_outputs():
    """'drift' debe depender de los pkl del stage 'train'."""
    content = _load_dvc_yaml()
    drift_deps = content["stages"]["drift"].get("deps", [])
    expected = {
        "models/pipeline_salidas.pkl",
        "models/pipeline_llegadas.pkl",
    }
    assert expected.issubset(set(drift_deps)), (
        f"'drift' no depende de los modelos generados por 'train': "
        f"{expected - set(drift_deps)}"
    )


def test_drift_depends_on_features_panel():
    """'drift' debe depender del panel de features v3."""
    content = _load_dvc_yaml()
    drift_deps = content["stages"]["drift"].get("deps", [])
    assert "data/processed/demand_features_v3.parquet" in drift_deps, (
        "'drift' necesita el panel v3 para simular los escenarios de drift."
    )


def test_drift_has_params():
    """'drift' debe referenciar la seccion 'drift' de params.yaml."""
    content = _load_dvc_yaml()
    drift_params = content["stages"]["drift"].get("params", [])
    param_keys = []
    for item in drift_params:
        if isinstance(item, dict):
            for vals in item.values():
                param_keys.extend(vals if vals else [])
    assert "drift" in param_keys, (
        "El stage 'drift' debe referenciar la seccion 'drift' de params.yaml "
        "(umbrales PSI, escenarios, etc.)."
    )


def test_drift_produces_summary_csv():
    """'drift' debe producir reports/drift/resumen_drift.csv."""
    content = _load_dvc_yaml()
    drift_outs = content["stages"]["drift"].get("outs", [])
    out_paths = []
    for item in drift_outs:
        if isinstance(item, str):
            out_paths.append(item)
        elif isinstance(item, dict):
            out_paths.extend(item.keys())
    assert "reports/drift/resumen_drift.csv" in out_paths, (
        "'drift' debe declarar 'reports/drift/resumen_drift.csv' como output."
    )
