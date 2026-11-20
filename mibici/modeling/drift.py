"""Simulacion de data drift y deteccion de perdida de desempeno (Entrega final).

Pregunta que responde: si los datos de entrada cambian despues del despliegue,
¿el monitoreo lo detecta y que accion se debe tomar?

Flujo:
  1. Conjunto de referencia: el primer mes posterior al entrenamiento
     (params.yaml > validacion.primer_mes_prueba), sin alteraciones.
  2. Escenarios sinteticos (params.yaml > drift.escenarios), generados a partir de
     la referencia con una semilla fija:
       - demanda:     la demanda real cambia por un factor y las features no
                      (cambio en la relacion entrada-demanda).
       - features:    columnas de entrada multiplicadas por un factor
                      (p. ej. un error en el pipeline que duplica conteos).
       - faltantes:   una fraccion de valores de algunas columnas se pierde.
       - estacional:  cambia el mes y la demanda (temporada de vacaciones).
  3. Deteccion en dos niveles:
       - Datos (no requiere etiquetas): PSI por columna y aumento de valores nulos.
       - Desempeno (cuando llegan los valores reales, un dia despues): MAE del
         modelo contra el MAE de referencia y contra la linea base
         (promedio historico, ver validation.py).
  4. Cada escenario recibe un estado (OK, AVISO, ALERTA), los motivos y una
     accion propuesta. Resultados en reports/drift/, graficas en
     reports/figures/ y un run por escenario en MLflow (experimento de monitoreo).

Uso:
    python -m mibici.modeling.drift                   # modelo @champion del Model Registry
    python -m mibici.modeling.drift --source local    # pipelines de models/*.pkl
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # sin pantalla

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import FIGURES_DIR, REPORTS_DIR, load_params
from mibici.modeling.predict import Source, load_champion, load_pipeline
from mibici.modeling.train import ALL_FEATURES
from mibici.modeling.validation import HistoricalMean, net_balance_metrics, regression_metrics
from mibici.versioning import TRACKING_URI

app = typer.Typer(help="Simulacion de data drift y deteccion de perdida de desempeno.")

DRIFT_DIR = REPORTS_DIR / "drift"
ESTADOS = ["OK", "AVISO", "ALERTA"]


# ---------------------------------------------------------------------------
# Generacion de escenarios sinteticos
# ---------------------------------------------------------------------------
def make_scenario(
    reference: pd.DataFrame, spec: dict, targets: list[str], seed: int
) -> pd.DataFrame:
    """Devuelve una copia de la referencia con la alteracion descrita en `spec`.

    Los conteos escalados se redondean de forma estocastica (el valor esperado es
    exactamente conteo x factor) para que sigan siendo enteros no negativos sin
    agregar ruido extra: con factor 1 el escenario es identico a la referencia.
    """
    rng = np.random.default_rng(seed)
    df = reference.copy()
    kind = spec["tipo"]

    if kind in ("demanda", "estacional"):
        for t in targets:
            scaled = df[t].to_numpy(dtype=float) * spec["factor"]
            base = np.floor(scaled)
            df[t] = (base + (rng.random(len(df)) < scaled - base)).astype(int)
        if kind == "estacional":
            df["mes"] = np.int8(spec["mes"])
    elif kind == "features":
        for col in spec["columnas"]:
            df[col] = df[col] * spec["factor"]
    elif kind == "faltantes":
        for col in spec["columnas"]:
            mask = rng.random(len(df)) < spec["fraccion"]
            df.loc[mask, col] = np.nan
    else:
        raise ValueError(f"Tipo de escenario desconocido: {kind}")
    return df


# ---------------------------------------------------------------------------
# Deteccion de drift en los datos de entrada
# ---------------------------------------------------------------------------
def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index entre dos muestras de una columna numerica.

    Los cortes son cuantiles de la referencia (los repetidos se unen, util en
    columnas discretas como la hora). Los nulos se ignoran aqui y se miden
    aparte con `missing_increase`. Regla usual: < 0.1 estable, 0.1-0.25 cambio
    moderado, > 0.25 cambio significativo.
    """
    ref = pd.Series(reference, dtype=float).dropna().to_numpy()
    cur = pd.Series(current, dtype=float).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # columna constante en la referencia: dos grupos, igual o distinto
        same = float(np.mean(np.isclose(cur, edges[0])))
        p, q = np.array([1.0, 0.0]), np.array([same, 1.0 - same])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def missing_increase(reference, current) -> float:
    """Aumento en la fraccion de valores nulos respecto de la referencia."""
    return float(pd.Series(current).isna().mean() - pd.Series(reference).isna().mean())


def data_drift(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], bins: int
) -> pd.DataFrame:
    """PSI y aumento de nulos por columna monitoreada."""
    rows = [
        {
            "columna": col,
            "psi": psi(reference[col], current[col], bins),
            "aumento_nulos": missing_increase(reference[col], current[col]),
        }
        for col in columns
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Desempeno y reglas de alerta
# ---------------------------------------------------------------------------
def performance(
    df: pd.DataFrame, preds: dict[str, np.ndarray], targets: list[str], peak_hours
) -> dict:
    """MAE, RMSE, MAE en hora pico y sesgo por objetivo, y MAE del balance neto.

    El sesgo es (pronostico total - demanda total) / demanda total: positivo si el
    modelo sobreestima y negativo si subestima. A diferencia del MAE, no baja solo
    porque la demanda baje, por eso detecta cambios de nivel en cualquier direccion.
    """
    out = {}
    for t in targets:
        m = regression_metrics(df[t], preds[t], df["hora"], peak_hours)
        out.update({f"{t}_{k}": v for k, v in m.items() if k in ("mae", "rmse", "mae_pico")})
        total = float(np.sum(df[t]))
        out[f"{t}_sesgo"] = float(np.sum(preds[t]) - total) / total if total else float("nan")
    if {"salidas", "llegadas"} <= set(targets):
        b = net_balance_metrics(
            df["salidas"],
            df["llegadas"],
            preds["salidas"],
            preds["llegadas"],
            df["hora"],
            peak_hours,
        )
        out["balance_mae"] = b["balance_mae"]
    return out


def evaluate_alerts(
    drift_table: pd.DataFrame,
    perf: dict,
    perf_ref: dict,
    baseline_perf: dict,
    targets: list[str],
    criteria: list[str],
    thresholds: dict,
) -> dict:
    """Aplica los umbrales de params.yaml y propone una accion.

    Niveles por regla (el estado final es el peor):
      - PSI de alguna columna     >= psi_aviso   -> AVISO;  >= psi_alerta   -> ALERTA
      - Aumento de nulos          >= faltantes_alerta -> ALERTA
      - |Cambio en el sesgo|      >= sesgo_aviso -> AVISO;  >= sesgo_alerta -> ALERTA
      - MAE / MAE de referencia   >= mae_aviso   -> AVISO;  >= mae_alerta   -> ALERTA
      - MAE del modelo > MAE de la linea base x (1 + margen_linea_base) -> ALERTA
    Las reglas de datos no requieren conocer la demanda real; las de desempeno se
    evaluan cuando llegan los valores reales (un dia despues del pronostico).
    """
    level, reasons = 0, []

    def raise_to(new_level: int, reason: str):
        nonlocal level
        level = max(level, new_level)
        reasons.append(reason)

    # --- Datos de entrada ---
    valid = drift_table.dropna(subset=["psi"])
    psi_max = float(valid["psi"].max()) if len(valid) else 0.0
    worst = valid.loc[valid["psi"].idxmax(), "columna"] if len(valid) else "-"
    if psi_max >= thresholds["psi_alerta"]:
        raise_to(2, f"PSI {psi_max:.2f} en {worst}")
    elif psi_max >= thresholds["psi_aviso"]:
        raise_to(1, f"PSI {psi_max:.2f} en {worst}")

    missing = drift_table[drift_table["aumento_nulos"] >= thresholds["faltantes_alerta"]]
    for row in missing.itertuples():
        raise_to(2, f"{row.aumento_nulos:.0%} de nulos nuevos en {row.columna}")
    data_level = level

    # --- Desempeno ---
    bias = {t: perf[f"{t}_sesgo"] - perf_ref[f"{t}_sesgo"] for t in targets}
    for t, b in bias.items():
        direction = "sobreestima" if b > 0 else "subestima"
        if abs(b) >= thresholds["sesgo_alerta"]:
            raise_to(2, f"{t}: {direction} {abs(b):.0%}")
        elif abs(b) >= thresholds["sesgo_aviso"]:
            raise_to(1, f"{t}: {direction} {abs(b):.0%}")

    ratios = {c: perf[f"{c}_mae"] / perf_ref[f"{c}_mae"] for c in criteria}
    for c, r in ratios.items():
        if r >= thresholds["mae_alerta"]:
            raise_to(2, f"MAE de {c} {r - 1:+.0%}")
        elif r >= thresholds["mae_aviso"]:
            raise_to(1, f"MAE de {c} {r - 1:+.0%}")

    margin = 1 + thresholds["margen_linea_base"]
    worse_than_baseline = [
        c for c in criteria if perf[f"{c}_mae"] > baseline_perf[f"{c}_mae"] * margin
    ]
    for c in worse_than_baseline:
        raise_to(2, f"{c} peor que la linea base")

    # --- Accion propuesta (la primera que aplique) ---
    if level == 0:
        action = "Sin accion: seguir monitoreando."
    elif data_level == 2:
        action = "Revisar el pipeline de features: cambiaron los datos de entrada."
    elif worse_than_baseline:
        action = "Usar la linea base mientras se reentrena con datos recientes."
    elif level == 2:
        action = "Reentrenar con datos recientes: cambio el nivel de la demanda."
    else:
        action = "Revisar en el siguiente ciclo de monitoreo."

    return {
        "estado": ESTADOS[level],
        "motivos": "; ".join(reasons) if reasons else "-",
        "accion": action,
        "psi_max": float(psi_max),
        **{f"cambio_sesgo_{t}": float(b) for t, b in bias.items()},
        **{f"ratio_mae_{c}": float(r) for c, r in ratios.items()},
    }


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------
def _bars(ax, labels, series: dict[str, np.ndarray]):
    """Barras agrupadas por escenario, una serie por objetivo o criterio."""
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + i * width, values, width, label=name)
    ax.set_xticks(x + width * (len(series) - 1) / 2, labels, rotation=15)


def _thresholds(ax, low: float, high: float, symmetric: bool = False):
    for value, style, label in ((low, "--", "umbral aviso"), (high, "-", "umbral alerta")):
        ax.axhline(value, color="0.3", linestyle=style, linewidth=1, label=label)
        if symmetric:
            ax.axhline(-value, color="0.3", linestyle=style, linewidth=1)


def plot_results(
    summary: pd.DataFrame,
    targets: list[str],
    criteria: list[str],
    thresholds: dict,
    out_dir: Path,
    psi_cap: float = 2.0,
) -> list[Path]:
    """Tres figuras por escenario: PSI maximo, cambio en el sesgo y cambio en el MAE."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["escenario"]
    paths = []

    fig, ax = plt.subplots(figsize=(10, 5))
    shown = summary["psi_max"].clip(upper=psi_cap)
    ax.bar(labels, shown, color="#5d6771")
    for x, (real, bar) in enumerate(zip(summary["psi_max"], shown)):
        ax.text(x, bar, f"{real:.2f}", ha="center", va="bottom")
    _thresholds(ax, thresholds["psi_aviso"], thresholds["psi_alerta"])
    ax.set_ylabel(f"PSI maximo entre columnas (recortado en {psi_cap})")
    ax.set_title("Drift en los datos de entrada (no requiere la demanda real)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    paths.append(out_dir / "drift_psi.png")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    _bars(ax2, labels, {t: summary[f"cambio_sesgo_{t}"] * 100 for t in targets})
    _thresholds(ax2, thresholds["sesgo_aviso"] * 100, thresholds["sesgo_alerta"] * 100, True)
    ax2.axhline(0, color="0.6", linewidth=0.8)
    ax2.set_ylabel("Cambio en el sesgo vs referencia (%)")
    ax2.set_title("Sobreestimacion (+) o subestimacion (-) de la demanda")
    ax2.legend()
    paths.append(out_dir / "drift_sesgo.png")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    _bars(ax3, labels, {c: (summary[f"ratio_mae_{c}"] - 1) * 100 for c in criteria})
    _thresholds(ax3, (thresholds["mae_aviso"] - 1) * 100, (thresholds["mae_alerta"] - 1) * 100)
    ax3.set_ylabel("Cambio en el MAE vs referencia (%)")
    ax3.set_title("Cambio en el error absoluto por escenario")
    ax3.legend()
    paths.append(out_dir / "drift_mae.png")

    for figure, path in zip((fig, fig2, fig3), paths):
        figure.tight_layout()
        figure.savefig(path, dpi=120)
        plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    source: Source = typer.Option(
        Source.registry, help="registry: @champion del Model Registry; local: models/*.pkl"
    ),
) -> None:
    """Genera los escenarios, mide el drift y registra los resultados en MLflow."""
    params = load_params()
    cfg = params["drift"]
    targets = params["problema"]["objetivos"]
    peak = params["problema"]["horas_pico"]
    criteria = params["promocion"]["criterios"]
    thresholds = cfg["umbrales"]
    seed = params.get("seed", 42)
    time_col = params["validacion"]["columna_tiempo"]

    ref_start = pd.Timestamp(params["validacion"]["primer_mes_prueba"] + "-01")
    ref_end = ref_start + pd.DateOffset(months=params["validacion"]["meses_prueba"])
    train_start = ref_start - pd.DateOffset(months=params["validacion"]["meses_entrenamiento"])

    keys = list(params["lineas_base"]["llaves_promedio"])
    columns = sorted({time_col, *keys, *targets, *ALL_FEATURES})
    df = pd.read_parquet(params["datos"]["features"], columns=columns)
    reference = df[(df[time_col] >= ref_start) & (df[time_col] < ref_end)].reset_index(drop=True)
    history = df[(df[time_col] >= train_start) & (df[time_col] < ref_start)]
    del df
    logger.info(f"Referencia {ref_start:%Y-%m}: {len(reference):,} filas")

    models = {
        t: load_champion(params, t) if source is Source.registry else load_pipeline(t)
        for t in targets
    }
    baseline = {t: HistoricalMean(tuple(keys)).fit(history[keys], history[t]) for t in targets}
    del history

    def predict_all(frame):
        model = {t: np.clip(models[t].predict(frame[ALL_FEATURES]), 0, None) for t in targets}
        base = {t: baseline[t].predict(frame[keys]) for t in targets}
        return performance(frame, model, targets, peak), performance(frame, base, targets, peak)

    perf_ref, _ = predict_all(reference)
    logger.info(
        "Referencia: " + ", ".join(f"{c} MAE={perf_ref[f'{c}_mae']:.3f}" for c in criteria)
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(cfg["experimento"])
    rows, psi_tables = [], []
    with mlflow.start_run(run_name="simulacion_drift"):
        mlflow.set_tags({"tipo": "monitoreo", "modelo": source.value, "dataset_version": "v3"})
        mlflow.log_params(
            {
                "mes_referencia": f"{ref_start:%Y-%m}",
                "seed": seed,
                "fuente_modelo": source.value,
                **{f"umbral_{k}": v for k, v in thresholds.items()},
            }
        )
        mlflow.log_metrics({f"referencia_{k}": v for k, v in perf_ref.items()})

        for i, (name, spec) in enumerate(cfg["escenarios"].items()):
            current = make_scenario(reference, spec, targets, seed + i)
            table = data_drift(reference, current, cfg["columnas_monitoreo"], cfg["bins_psi"])
            perf, perf_base = predict_all(current)
            result = evaluate_alerts(
                table, perf, perf_ref, perf_base, targets, criteria, thresholds
            )
            row = {
                "escenario": name,
                "tipo": spec["tipo"],
                **result,
                **{f"{c}_mae": perf[f"{c}_mae"] for c in criteria},
                **{f"{c}_mae_linea_base": perf_base[f"{c}_mae"] for c in criteria},
            }
            rows.append(row)
            psi_tables.append(table.assign(escenario=name))
            logger.info(f"[{name}] {result['estado']}: {result['motivos']}")

            with mlflow.start_run(run_name=f"drift_{name}", nested=True):
                mlflow.set_tags(
                    {
                        "tipo": "monitoreo",
                        "escenario": name,
                        "estado": result["estado"],
                        "accion": result["accion"],
                    }
                )
                mlflow.log_params({f"escenario_{k}": v for k, v in spec.items()})
                mlflow.log_metrics(
                    {
                        **{k: v for k, v in perf.items()},
                        **{f"linea_base_{k}": v for k, v in perf_base.items()},
                        "psi_max": result["psi_max"],
                        **{
                            k: v
                            for k, v in result.items()
                            if k.startswith(("ratio_mae_", "cambio_sesgo_"))
                        },
                    }
                )
                mlflow.log_table(table, f"psi_{name}.json")

        summary = pd.DataFrame(rows)
        DRIFT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(DRIFT_DIR / "resumen_drift.csv", index=False)
        pd.concat(psi_tables).to_csv(DRIFT_DIR / "psi_por_columna.csv", index=False)
        for path in plot_results(summary, targets, criteria, thresholds, FIGURES_DIR):
            mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(DRIFT_DIR / "resumen_drift.csv"))
        mlflow.log_artifact(str(DRIFT_DIR / "psi_por_columna.csv"))

    logger.success(f"Resumen en {DRIFT_DIR / 'resumen_drift.csv'}")
    print(summary[["escenario", "estado", "motivos", "accion"]].to_string(index=False))


if __name__ == "__main__":
    app()
