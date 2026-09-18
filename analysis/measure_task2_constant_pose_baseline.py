"""¿Cuánto puntúa en Task 2 predecir SIEMPRE una matriz constante?

Pregunta concreta (ver experiments/95-t2-constant-pose/INFORME.md): el
leaderboard tiene 7 equipos resolviendo Task 2 mientras 5 auditorías internas
concluyen que no hay correspondencia visual explotable (IoU vasculatura
0.015-0.07). Hipótesis: el score alto puede venir de que las poses estén
concentradas por escenario y una constante bien elegida ya puntúe, sin "ver"
nada.

Este script NO carga imágenes ni modelos. Solo lee las matrices Ground Truth
de Task 2 (train: 1214 casos / 10 escenarios; Mock Test: 5 casos / 1
escenario no visto en train) y mide, con la métrica OFICIAL exacta
(`fido.geometry.corner_auc`, réplica verificada de
`vendor/fido/Codabench Bundle/scoring_program/scoring_registration.py`):

  1. Constante global (mediana de los 1214 casos, evaluada en los 1214).
  2. Constante por escenario en oráculo (mediana de cada escenario, evaluada
     en ESE escenario).
  3. Leave-one-scenario-out (mediana de 9 escenarios, evaluada en el 10º) —
     la medida honesta.
  4. Sobre el Mock Test (5 casos, escenario nunca visto en train): predecir
     la constante global de train.
  5. Repite 1-3 con media y moda además de mediana.

Parametrización de 4 DOF y descomposición/composición: `fido.geometry`
(`decompose_similarity` / `compose_similarity`, reflexión fija `reflect=True`,
ver docstring — verificado en el 100% de los 1214 casos de train). La
"mediana de una pose" se calcula por parámetro, NO elemento a elemento sobre
la matriz 3x3: (tx, ty, escala) con mediana/media/moda estándar; theta es
circular (ver `circular_median` / `circular_mean` / `circular_mode` abajo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fido.geometry import compose_similarity, corner_auc, corner_error, decompose_similarity  # noqa: E402

TRAIN_ROOT = Path("data/_annotations/Task 2")
MOCK_ROOT = Path("data/Mock Test/Task 2")


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------

def load_cases(root: Path, numerical_subdir: str | None = None):
    """Recorre `root`/Scenario_XX[/numerical_subdir]/*.json y devuelve
    (case_id, scenario, matrix) para cada caso con Ground Truth Task 2.

    `case_id` sigue exactamente la convención del scorer oficial
    (`case_id.rsplit("_", 1)` -> scenario_name, frame_id), así que aquí se
    construye como f"{scenario}_{frame_stem}".
    """
    cases = []
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        search_dir = scenario_dir / numerical_subdir if numerical_subdir else scenario_dir
        if not search_dir.is_dir():
            continue
        for path in sorted(search_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            gt = (data.get("Ground Truth") or {}).get("Task 2")
            if gt is None:
                continue
            matrix = np.asarray(gt, dtype=np.float64)
            if matrix.shape != (3, 3):
                continue
            case_id = f"{scenario_dir.name}_{path.stem}"
            cases.append((case_id, scenario_dir.name, matrix))
    return cases


# --------------------------------------------------------------------------
# Estadísticos de pose (por parámetro, theta circular)
# --------------------------------------------------------------------------

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def circular_mean(theta):
    return float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))


def _unwrap_around_mean(theta):
    """Representante lineal de cada ángulo, centrado en la dirección media
    (dentro de (mean-pi, mean+pi]). Válido mientras la muestra no cubra más
    de media vuelta — cierto aquí (ver INFORME.md, sd de theta por escenario)."""
    mean_dir = circular_mean(theta)
    return mean_dir, mean_dir + wrap_to_pi(theta - mean_dir)


def circular_median(theta):
    """Mediana circular vía desenvolvimiento alrededor de la media circular:
    se lleva cada ángulo a su representante lineal más cercano a la
    dirección media, se toma la mediana lineal, y se vuelve a envolver a
    (-pi, pi]. Equivale a la mediana circular estándar para muestras
    unimodales que no cubren más de media vuelta (caso de Task 2, ver
    INFORME.md)."""
    _, unwrapped = _unwrap_around_mean(theta)
    return float(wrap_to_pi(np.median(unwrapped)))


def _kde_mode_1d(values, n_grid=2000):
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=np.float64)
    if np.ptp(values) < 1e-9:  # todos los valores (casi) iguales
        return float(np.median(values))
    kde = gaussian_kde(values)
    lo, hi = values.min(), values.max()
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    density = kde(grid)
    return float(grid[np.argmax(density)])


def circular_mode(theta):
    """Moda de un ángulo: KDE 1D sobre el desenvolvimiento alrededor de la
    media circular (mismo truco que `circular_median`), luego se envuelve."""
    _, unwrapped = _unwrap_around_mean(theta)
    return float(wrap_to_pi(_kde_mode_1d(unwrapped)))


def pose_stat(matrices: np.ndarray, method: str) -> dict:
    """Estadístico de pose por parámetro sobre un batch de matrices (N,3,3).
    Devuelve el dict de parámetros que espera `compose_similarity`."""
    params = decompose_similarity(matrices, reflect=True)
    theta = np.arctan2(params["sin_theta"], params["cos_theta"])
    tx, ty, scale = params["tx"], params["ty"], params["scale"]

    if method == "median":
        tx_s, ty_s, scale_s = np.median(tx), np.median(ty), np.median(scale)
        theta_s = circular_median(theta)
    elif method == "mean":
        tx_s, ty_s, scale_s = np.mean(tx), np.mean(ty), np.mean(scale)
        theta_s = circular_mean(theta)
    elif method == "mode":
        tx_s, ty_s, scale_s = _kde_mode_1d(tx), _kde_mode_1d(ty), _kde_mode_1d(scale)
        theta_s = circular_mode(theta)
    else:
        raise ValueError(f"unknown method {method!r}")

    return dict(tx=float(tx_s), ty=float(ty_s),
                cos_theta=float(np.cos(theta_s)), sin_theta=float(np.sin(theta_s)),
                scale=float(scale_s))


def stat_to_matrix(params: dict) -> np.ndarray:
    return np.asarray(compose_similarity(reflect=True, **params), dtype=np.float64)


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------

THRESHOLDS_PX = (10, 25, 50, 100, 150, 200)


def eval_constant(pred_matrix: np.ndarray, gt_matrices: np.ndarray) -> dict:
    pred_batch = np.broadcast_to(pred_matrix, gt_matrices.shape)
    errors = np.asarray(corner_error(pred_batch, gt_matrices), dtype=np.float64)
    frac_below = {str(t): float(np.mean(errors <= t)) for t in THRESHOLDS_PX}
    return dict(n=int(len(errors)), mean_error=float(np.mean(errors)),
                median_error=float(np.median(errors)), auc=float(corner_auc(errors)),
                frac_below_threshold=frac_below, errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                         default=Path("experiments/95-t2-constant-pose/results.json"))
    args = parser.parse_args()

    train_cases = load_cases(TRAIN_ROOT)
    mock_cases = load_cases(MOCK_ROOT, numerical_subdir="Numerical")
    assert len(train_cases) == 1214, f"esperaba 1214 casos de train, encontré {len(train_cases)}"

    ids = [c[0] for c in train_cases]
    scenarios = np.asarray([c[1] for c in train_cases])
    matrices = np.stack([c[2] for c in train_cases])
    scenario_names = sorted(set(scenarios.tolist()))

    mock_ids = [c[0] for c in mock_cases]
    mock_scenarios = sorted(set(c[1] for c in mock_cases))
    mock_matrices = np.stack([c[2] for c in mock_cases]) if mock_cases else np.zeros((0, 3, 3))

    results: dict = {
        "n_train_cases": len(train_cases),
        "n_scenarios": len(scenario_names),
        "scenario_names": scenario_names,
        "mock_scenarios": mock_scenarios,
        "n_mock_cases": len(mock_cases),
        "methods": {},
    }

    for method in ("median", "mean", "mode"):
        method_result: dict = {}

        # 1) Constante global: stat de los 1214, evaluada en los 1214.
        global_params = pose_stat(matrices, method)
        global_matrix = stat_to_matrix(global_params)
        global_eval = eval_constant(global_matrix, matrices)
        method_result["global_in_sample"] = {
            "params": global_params,
            **{k: v for k, v in global_eval.items() if k != "errors"},
        }

        # 2) Constante por escenario, oráculo: stat de CADA escenario evaluada
        #    en ESE escenario.
        per_scenario_oracle = {}
        oracle_all_errors = []
        for scenario in scenario_names:
            mask = scenarios == scenario
            scen_matrices = matrices[mask]
            scen_params = pose_stat(scen_matrices, method)
            scen_matrix = stat_to_matrix(scen_params)
            scen_eval = eval_constant(scen_matrix, scen_matrices)
            oracle_all_errors.append(scen_eval["errors"])
            per_scenario_oracle[scenario] = {
                "params": scen_params,
                **{k: v for k, v in scen_eval.items() if k != "errors"},
            }
        pooled_oracle_errors = np.concatenate(oracle_all_errors)
        method_result["per_scenario_oracle"] = {
            "per_scenario": per_scenario_oracle,
            "pooled": {
                "n": int(len(pooled_oracle_errors)),
                "mean_error": float(np.mean(pooled_oracle_errors)),
                "median_error": float(np.median(pooled_oracle_errors)),
                "auc": float(corner_auc(pooled_oracle_errors)),
                "frac_below_threshold": {str(t): float(np.mean(pooled_oracle_errors <= t))
                                         for t in THRESHOLDS_PX},
            },
        }

        # 3) Leave-one-scenario-out: stat de los otros 9, evaluada en el 10.
        per_scenario_loso = {}
        loso_all_errors = []
        for held_out in scenario_names:
            train_mask = scenarios != held_out
            eval_mask = scenarios == held_out
            loso_params = pose_stat(matrices[train_mask], method)
            loso_matrix = stat_to_matrix(loso_params)
            loso_eval = eval_constant(loso_matrix, matrices[eval_mask])
            loso_all_errors.append(loso_eval["errors"])
            per_scenario_loso[held_out] = {
                "params": loso_params,
                **{k: v for k, v in loso_eval.items() if k != "errors"},
            }
        pooled_loso_errors = np.concatenate(loso_all_errors)
        method_result["leave_one_scenario_out"] = {
            "per_scenario": per_scenario_loso,
            "pooled": {
                "n": int(len(pooled_loso_errors)),
                "mean_error": float(np.mean(pooled_loso_errors)),
                "median_error": float(np.median(pooled_loso_errors)),
                "auc": float(corner_auc(pooled_loso_errors)),
                "frac_below_threshold": {str(t): float(np.mean(pooled_loso_errors <= t))
                                         for t in THRESHOLDS_PX},
            },
        }

        # 4) Mock Test: constante global de train (idéntica a la de 1), medida
        #    sobre los 5 casos del Mock Test (escenario nunca visto en train).
        if len(mock_cases):
            mock_eval = eval_constant(global_matrix, mock_matrices)
            method_result["mock_test"] = {
                "case_ids": mock_ids,
                "scenario": mock_scenarios,
                "per_case_error": mock_eval["errors"].tolist(),
                **{k: v for k, v in mock_eval.items() if k != "errors"},
            }

        results["methods"][method] = method_result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- impresión resumida ----
    print(f"train cases: {len(train_cases)}  scenarios: {scenario_names}")
    print(f"mock cases: {len(mock_cases)}  mock scenarios: {mock_scenarios}")
    for method in ("median", "mean", "mode"):
        r = results["methods"][method]
        print(f"\n===== método: {method} =====")
        g = r["global_in_sample"]
        print(f"  [1] global in-sample (1214 -> 1214):        "
              f"AUC={g['auc']:.4f}  mean_err={g['mean_error']:.2f}px  median_err={g['median_error']:.2f}px")
        po = r["per_scenario_oracle"]["pooled"]
        print(f"  [2] per-scenario oracle pooled:              "
              f"AUC={po['auc']:.4f}  mean_err={po['mean_error']:.2f}px  median_err={po['median_error']:.2f}px")
        pl = r["leave_one_scenario_out"]["pooled"]
        print(f"  [3] LOSO pooled (HONESTO):                   "
              f"AUC={pl['auc']:.4f}  mean_err={pl['mean_error']:.2f}px  median_err={pl['median_error']:.2f}px")
        if "mock_test" in r:
            m = r["mock_test"]
            print(f"  [4] Mock Test (5 casos, escenario no visto): "
                  f"AUC={m['auc']:.4f}  mean_err={m['mean_error']:.2f}px  median_err={m['median_error']:.2f}px")
        print("  -- desglose LOSO por escenario --")
        for scen, d in r["leave_one_scenario_out"]["per_scenario"].items():
            print(f"     {scen}: n={d['n']:<4} AUC={d['auc']:.4f}  mean_err={d['mean_error']:.2f}px")

    print(f"\nresultados completos: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
