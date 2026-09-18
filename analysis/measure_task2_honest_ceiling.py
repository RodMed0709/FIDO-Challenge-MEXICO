"""T2-101 -- techo honesto end-to-end de la via geometrica de Task 2, SIN
ningun GT en tiempo de inferencia (ni `iOCT.Rotation` ni el fundus del propio
escenario de test).

Contexto (T2-98/99/100, ver INFORMEs en `experiments/98-t2-pose-estimation/`,
`experiments/99-t2-scenario-heterogeneity/`, `experiments/100-t2-coefficients-
from-image/`): el modelo geometrico gnomonico de primer orden
(`analysis/derive_task2_geometry.py`) da AUC LOSO=0.1512 con coeficientes
globales y `iOCT.Rotation` EXACTO (leido del GT). T2-100 mejoro a AUC=0.3201
recalibrando los 8 coeficientes de escala/traslacion por escenario a partir de
una feature del fundus (`profile_edge_r50`) -- pero ese 0.3201 SIGUE usando
`iOCT.Rotation` GT del caso de test para computar `theta`, `gx`, `gy` en
`dtg.predict_matrix(ioct_q[test_idx], ...)`. En produccion (`inference()` real)
no existe ese GT: solo hay `oct_volume` y `opmi_image`.

Este script mide el AUC LOSO real reemplazando `iOCT.Rotation` GT por una
prediccion OOF entrenada SOLO con features del volumen OCT (reusa el cache de
`experiments/96-t2-oct-self-localisation/features_cache.npz`, mismo target que
T2-98 seccion 1: los 4 componentes del cuaternion, R2=0.4834 con
compact/random_forest) -- todo bajo leave-one-scenario-out anidado real: el
regresor de rotacion, la calibracion del boresight, el ajuste global de
theta0/coeficientes y el predictor imagen->coeficientes se entrenan TODOS solo
con los 9 escenarios de train de cada fold externo.

Ejecutar:
    python analysis/measure_task2_honest_ceiling.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from fido.geometry import corner_auc, corner_error  # noqa: E402
from measure_task2_oct_self_localisation import group_kfold_splits, N_SPLITS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "derive_task2_geometry", REPO_ROOT / "analysis" / "derive_task2_geometry.py"
)
dtg = importlib.util.module_from_spec(_spec)
sys.modules["derive_task2_geometry"] = dtg
_spec.loader.exec_module(dtg)

_spec2 = importlib.util.spec_from_file_location(
    "estimate_task2_coefficients_from_image", REPO_ROOT / "analysis" / "estimate_task2_coefficients_from_image.py"
)
eci = importlib.util.module_from_spec(_spec2)
sys.modules["estimate_task2_coefficients_from_image"] = eci
_spec2.loader.exec_module(eci)

ANNOTATIONS_ROOT = REPO_ROOT / "data" / "_annotations" / "Task 2"
OCT_CACHE_PATH = REPO_ROOT / "experiments" / "96-t2-oct-self-localisation" / "features_cache.npz"
IMAGE_CACHE_PATH = REPO_ROOT / "experiments" / "100-t2-coefficients-from-image" / "image_features_cache.npz"
OUTPUT_DIR = REPO_ROOT / "experiments" / "101-t2-honest-ceiling"

PRIMARY_IMAGE_FEATURE = ["profile_edge_r50"]  # ganador de T2-100 (AUC=0.3201 con GT-rotation)


def load_oct_cache(cache_path: Path) -> dict:
    payload = np.load(cache_path, allow_pickle=False)
    return {
        "X_compact": payload["X_compact"], "X_grid": payload["X_grid"],
        "gt_matrices": payload["gt_matrices"],
        "scenarios": payload["scenarios"].astype(str), "case_ids": payload["case_ids"].astype(str),
    }


def load_ioct_quats(case_ids: np.ndarray) -> np.ndarray:
    q = np.zeros((len(case_ids), 4), dtype=np.float64)
    for i, cid in enumerate(case_ids):
        scenario, frame_id = cid.split("/")
        data = json.loads((ANNOTATIONS_ROOT / scenario / f"{frame_id}.json").read_text(encoding="utf-8"))
        q[i] = data["iOCT Microscope"]["Spatial"]["Rotation"]
    return q


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angulo de rotacion (grados) entre dos cuaterniones unitarios xyzw,
    invariante al signo global (q y -q representan la misma rotacion)."""
    q1n = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2n = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    dot = np.abs(np.sum(q1n * q2n, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def rotation_regressor(seed: int):
    # compact/random_forest: mejor modelo verificado en T2-98 seccion 1
    # (R2_joint=0.4834 sobre los 4 componentes del cuaternion iOCT.Rotation).
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)


def run_honest_loso(oct_cache: dict, ioct_q: np.ndarray, X_img: np.ndarray,
                    image_feature_names: list[str], seed: int) -> dict:
    scenario = oct_cache["scenarios"]
    gt = oct_cache["gt_matrices"]
    case_ids = oct_cache["case_ids"]
    X_oct = oct_cache["X_compact"]  # "compact": el feature set ganador en T2-98 sec.1 (R2=0.4834, > full=0.4752)
    n = len(scenario)

    all_errors = np.zeros(n)
    per_scenario: dict = {}
    rotation_angular_error_deg = np.full(n, np.nan)

    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        train_scens = sorted(set(scenario[train_idx]))

        # --- 1. Rotacion iOCT desde el VOLUMEN OCT, entrenada solo en train ---
        reg = rotation_regressor(seed)
        reg.fit(X_oct[train_idx], ioct_q[train_idx])
        q_pred_test = reg.predict(X_oct[test_idx])
        q_pred_test = q_pred_test / np.linalg.norm(q_pred_test, axis=1, keepdims=True)
        rotation_angular_error_deg[test_idx] = quat_geodesic_deg(q_pred_test, ioct_q[test_idx])

        # --- 2. Boresight + modelo geometrico global, ajustados solo con train (GT ahi si disponible) ---
        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[train_idx])
        params_train = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)

        # --- 3. Coeficientes oraculo POR ESCENARIO DE TRAIN (GT, nunca del escenario de test) ---
        oracle_train = eci.oracle_coefficients_per_scenario(
            ioct_q[train_idx], gt[train_idx], scenario[train_idx], v0, e1, e2)

        # --- 4. Predictor imagen -> 8 coeficientes, ajustado solo con escenarios de train ---
        X_scen_mean = {}
        for s in train_scens + [scen]:
            idx_s = np.where(scenario == s)[0]
            X_scen_mean[s] = np.nanmean(X_img[idx_s], axis=0)
        Xtr = eci.build_design_matrix(X_scen_mean, train_scens, image_feature_names, eci.FEATURE_NAMES)
        Ytr = np.stack([oracle_train[s] for s in train_scens], axis=0)
        Xte = eci.build_design_matrix(X_scen_mean, [scen], image_feature_names, eci.FEATURE_NAMES)
        y_pred_coefs = eci.fit_predict_linear(Xtr, Ytr, Xte)[0]
        params_pred = eci.vector_to_params(y_pred_coefs, params_train["theta0"])

        # --- 5. Composicion final: rotacion PREDICHA (no GT) + coeficientes PREDICHOS (no GT) ---
        pred = dtg.predict_matrix(q_pred_test, v0, e1, e2, params_pred)
        errs = corner_error(pred, gt[test_idx])
        all_errors[test_idx] = errs
        per_scenario[scen] = dict(
            n=int(len(test_idx)), mean_error=float(errs.mean()), median_error=float(np.median(errs)),
            auc=float(corner_auc(errs)),
            rotation_angular_error_deg_mean=float(rotation_angular_error_deg[test_idx].mean()),
            rotation_angular_error_deg_median=float(np.median(rotation_angular_error_deg[test_idx])),
        )
        print(f"  scenario {scen}: n={len(test_idx):4d} mean_err={errs.mean():7.2f}px "
              f"AUC={corner_auc(errs):.4f} rot_err_mean={rotation_angular_error_deg[test_idx].mean():.2f}deg",
              flush=True)

    return dict(
        per_scenario=per_scenario,
        pooled_auc=float(corner_auc(all_errors)),
        pooled_mean_error=float(all_errors.mean()),
        pooled_median_error=float(np.median(all_errors)),
        rotation_angular_error_deg_mean=float(rotation_angular_error_deg.mean()),
        rotation_angular_error_deg_median=float(np.median(rotation_angular_error_deg)),
    )


def run_reference_gt_rotation_pred_coefs(oct_cache: dict, ioct_q: np.ndarray, X_img: np.ndarray,
                                         image_feature_names: list[str]) -> dict:
    """Reproduce EXACTAMENTE el 0.3201 de T2-100 (rotacion GT + coeficientes
    predichos desde imagen) como ancla de sanity check de este script."""
    scenario = oct_cache["scenarios"]
    gt = oct_cache["gt_matrices"]
    data = dict(ioct_q=ioct_q, gt=gt, scenario=scenario)
    r = eci.run_image_based_loso(data, X_img, image_feature_names)
    return dict(pooled_auc=r["pooled_auc"], pooled_mean_error=r["pooled_mean_error"])


def run_reference_pred_rotation_global_coefs(oct_cache: dict, ioct_q: np.ndarray, seed: int) -> dict:
    """Variante intermedia: rotacion PREDICHA desde el OCT + coeficientes
    GLOBALES (sin recalibrar por imagen) -- aisla el efecto de la rotacion
    predicha solo, sin mezclar con el efecto del predictor de coeficientes."""
    scenario = oct_cache["scenarios"]
    gt = oct_cache["gt_matrices"]
    X_oct = oct_cache["X_compact"]
    n = len(scenario)
    all_errors = np.zeros(n)
    per_scenario = {}
    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        reg = rotation_regressor(seed)
        reg.fit(X_oct[train_idx], ioct_q[train_idx])
        q_pred_test = reg.predict(X_oct[test_idx])
        q_pred_test = q_pred_test / np.linalg.norm(q_pred_test, axis=1, keepdims=True)
        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[train_idx])
        params_train = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)
        pred = dtg.predict_matrix(q_pred_test, v0, e1, e2, params_train)
        errs = corner_error(pred, gt[test_idx])
        all_errors[test_idx] = errs
        per_scenario[scen] = float(corner_auc(errs))
    return dict(per_scenario=per_scenario, pooled_auc=float(corner_auc(all_errors)),
                pooled_mean_error=float(all_errors.mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=== Cargando caches (OCT + imagen de fundus, de T2-96/98/100) ===", flush=True)
    oct_cache = load_oct_cache(OCT_CACHE_PATH)
    n = len(oct_cache["case_ids"])
    print(f"{n} casos, {len(set(oct_cache['scenarios']))} escenarios", flush=True)

    ioct_q = load_ioct_quats(oct_cache["case_ids"])

    img_payload = np.load(IMAGE_CACHE_PATH, allow_pickle=True)
    # El cache de OCT (T2-96) usa case_id="Scenario_01/00000"; el cache de
    # imagen (T2-100) usa case_id="01/00000" (formato de `dtg.load_dataset`).
    # Mismo caso, prefijo distinto -- se normaliza antes de comparar orden.
    oct_case_ids_short = [c.replace("Scenario_", "", 1) for c in oct_cache["case_ids"]]
    assert list(img_payload["case_id"]) == oct_case_ids_short, \
        "cache de imagen y cache de OCT desalineados -- revisar orden de casos"
    X_img = img_payload["X"]

    print("\n=== Referencia 1: reproducir T2-100 (rotacion GT + coef. predichos) ===", flush=True)
    ref_gt_rot = run_reference_gt_rotation_pred_coefs(oct_cache, ioct_q, X_img, PRIMARY_IMAGE_FEATURE)
    print(f"AUC={ref_gt_rot['pooled_auc']:.4f} mean_err={ref_gt_rot['pooled_mean_error']:.2f}px "
          f"[debe coincidir con 0.3201 de T2-100]", flush=True)

    print("\n=== Referencia 2: rotacion PREDICHA (OCT) + coeficientes GLOBALES (sin imagen) ===", flush=True)
    t0 = time.monotonic()
    ref_pred_rot_global = run_reference_pred_rotation_global_coefs(oct_cache, ioct_q, args.seed)
    print(f"AUC={ref_pred_rot_global['pooled_auc']:.4f} mean_err={ref_pred_rot_global['pooled_mean_error']:.2f}px "
          f"[{time.monotonic()-t0:.1f}s]", flush=True)

    print("\n=== RESULTADO PRINCIPAL: rotacion PREDICHA (OCT) + coeficientes PREDICHOS (fundus) -- "
          "SIN GT alguno en inferencia ===", flush=True)
    t0 = time.monotonic()
    honest = run_honest_loso(oct_cache, ioct_q, X_img, PRIMARY_IMAGE_FEATURE, args.seed)
    print(f"\nPOOLED HONESTO: AUC={honest['pooled_auc']:.4f}  mean_err={honest['pooled_mean_error']:.2f}px  "
          f"median_err={honest['pooled_median_error']:.2f}px  [{time.monotonic()-t0:.1f}s]", flush=True)
    print(f"Error angular de rotacion (OOF, OCT->iOCT.Rotation): "
          f"media={honest['rotation_angular_error_deg_mean']:.2f}deg "
          f"mediana={honest['rotation_angular_error_deg_median']:.2f}deg", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = dict(
        n_cases=n,
        reference_gt_rotation_pred_coefs_auc=ref_gt_rot["pooled_auc"],
        reference_gt_rotation_pred_coefs_mean_error=ref_gt_rot["pooled_mean_error"],
        reference_pred_rotation_global_coefs=ref_pred_rot_global,
        honest_end_to_end=honest,
        primary_image_feature=PRIMARY_IMAGE_FEATURE,
        rotation_model="compact+full OCT features / RandomForestRegressor(n_estimators=300, min_samples_leaf=3)",
        notes="honest_end_to_end usa SOLO oct_volume (features del volumen) + opmi_image "
              "(profile_edge_r50) -- ningun GT (iOCT.Rotation, coeficientes por escenario) se lee "
              "en el fold de test, en ningun paso.",
    )
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
