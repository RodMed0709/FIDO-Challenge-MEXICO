"""Test central de la hipotesis "el OCT sabe donde esta, sin mirar el fundus".

Pregunta: ¿el volumen OCT de Task 2, por si solo (sin el fundus), codifica
donde esta situado en la retina? Se mide extrayendo features baratas e
interpretables del volumen (perfil de espesor Ilm-Rpe por columna, densidad
de vasos, densidad de instrumento, area no capturada) y regresando (tx,ty),
(cos theta, sin theta) y escala del GT de Task 2 con GroupKFold por escenario
(10 folds = 10 escenarios, leave-one-scenario-out).

No usa el fundus en ningun punto del pipeline de features. Los datos viven en
`data/Task 2/Scenario_NN.zip` (9 de 10 escenarios, nunca extraidos a disco) o
`data/Task 2/Scenario_NN/` (extraido); este script lee ambos formatos sin
necesitar descomprimir nada nuevo. Las anotaciones GT ya estan extraidas en
`data/_annotations/Task 2/Scenario_NN/*.json`.

Uso:
    python analysis/measure_task2_oct_self_localisation.py \
        --output-dir experiments/96-t2-oct-self-localisation

Sub-pasos (ver docstrings de cada funcion):
  1. discover_cases           - enumera casos desde las anotaciones locales
  2. ScenarioVolumeReader      - lee Volume/Segmentation desde zip o disco
  3. extract_case_features     - features por caso (parte 1 y 3 del pedido)
  4. run_pipeline               - GroupKFold, modelos, AUC oficial, control shuffle
  5. write_report                - INFORME.md con hallazgos e interpretacion
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fido.eval_task2 import evaluate_task2_cases, require_task2_training_root  # noqa: E402
from fido.geometry import compose_similarity, decompose_similarity  # noqa: E402

LAYER_ILM = 1
LAYER_RPE = 2
VESSEL_CLASS = 3
INSTRUMENT_CLASSES = (8, 10, 11)
UNCAPTURED_CLASS = 14
GRID = 8  # downsample de los mapas 128x512 a GRID x GRID para el set "full"
N_SPLITS = 10  # 10 escenarios en Task 2 train -> leave-one-scenario-out exacto

COMPACT_FEATURE_NAMES = [
    "thickness_mean", "thickness_std", "thickness_min", "thickness_max",
    "thickness_p10", "thickness_p90", "thickness_range", "valid_fraction",
    "argmin_u", "argmin_v", "argmax_u", "argmax_v",
    "dip_centroid_u", "dip_centroid_v", "dip_weight_total",
    "missing_centroid_u", "missing_centroid_v", "missing_fraction",
    "thickness_grad_mean", "thickness_grad_std",
    "vessel_density_mean", "vessel_density_std", "vessel_centroid_u", "vessel_centroid_v",
    "instrument_present", "instrument_density_mean", "instrument_centroid_u", "instrument_centroid_v",
    "uncaptured_fraction", "uncaptured_centroid_u", "uncaptured_centroid_v",
    "ilm_pixel_total", "rpe_pixel_total",
]


# --------------------------------------------------------------------------
# 1. Descubrimiento de casos (solo anotaciones locales, sin tocar los zips)
# --------------------------------------------------------------------------

def discover_cases(annotations_root: Path, per_scenario_limit: int | None = None) -> list[dict]:
    """`per_scenario_limit` recorta CADA escenario por separado (no un slice
    global) para que un smoke test siga teniendo los 10 grupos que exige
    GroupKFold leave-one-scenario-out."""
    cases = []
    for scenario_dir in sorted(annotations_root.glob("Scenario_*")):
        json_paths = sorted(scenario_dir.glob("*.json"))
        if per_scenario_limit is not None:
            json_paths = json_paths[:per_scenario_limit]
        for json_path in json_paths:
            cases.append({
                "scenario": scenario_dir.name,
                "frame_id": json_path.stem,
                "json_path": json_path,
            })
    return cases


# --------------------------------------------------------------------------
# 2. Lector de volumen: extraido en disco o directo desde el zip (en memoria)
# --------------------------------------------------------------------------

class ScenarioVolumeReader:
    """Lee `Volume/<frame>/Segmentation/*.png` sin extraer los zips a disco.

    Si `data/Task 2/<scenario>/` ya existe (extraido), lee de ahi. Si no,
    abre `data/Task 2/<scenario>.zip` una vez y decodifica los PNG en
    memoria via `io.BytesIO`. Nunca escribe en disco.
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._zip_cache: dict[str, zipfile.ZipFile] = {}

    def _zip(self, scenario: str) -> zipfile.ZipFile:
        if scenario not in self._zip_cache:
            self._zip_cache[scenario] = zipfile.ZipFile(self.data_root / f"{scenario}.zip")
        return self._zip_cache[scenario]

    def read_segmentation(self, scenario: str, frame_id: str) -> np.ndarray:
        extracted_dir = (self.data_root / scenario / "iOCT Microscope" / "Volume"
                         / frame_id / "Segmentation")
        if extracted_dir.is_dir():
            files = sorted(extracted_dir.glob("*.png"))
            if not files:
                raise FileNotFoundError(f"sin PNG de segmentacion en {extracted_dir}")
            return np.stack([np.array(Image.open(f)) for f in files], axis=0)

        zf = self._zip(scenario)
        prefix = f"iOCT Microscope/Volume/{frame_id}/Segmentation/"
        names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png"))
        if not names:
            raise FileNotFoundError(f"sin entradas de segmentacion para {prefix} en {scenario}.zip")
        arrays = []
        for name in names:
            arrays.append(np.array(Image.open(io.BytesIO(zf.read(name)))))
        return np.stack(arrays, axis=0)

    def close(self) -> None:
        for zf in self._zip_cache.values():
            zf.close()


# --------------------------------------------------------------------------
# 3. Features por caso -- SOLO del volumen OCT, nunca del fundus
# --------------------------------------------------------------------------

def _normalized_coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    vv, uu = np.meshgrid(np.arange(h) / max(h - 1, 1), np.arange(w) / max(w - 1, 1), indexing="ij")
    return uu.astype(np.float32), vv.astype(np.float32)


def _weighted_centroid(weight_map: np.ndarray, uu: np.ndarray, vv: np.ndarray) -> tuple[float, float, float]:
    total = float(np.nansum(weight_map))
    if not np.isfinite(total) or total <= 1e-6:
        return -1.0, -1.0, 0.0
    wu = float(np.nansum(weight_map * uu) / total)
    wv = float(np.nansum(weight_map * vv) / total)
    return wu, wv, total


def compute_thickness_map(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """seg: (n_slices, depth, width) uint8. Devuelve (thickness, valid), ambos
    (n_slices, width). thickness = fila-media(Rpe) - fila-media(Ilm) por
    columna A-scan; NaN donde falta Ilm o Rpe en esa columna."""
    depth_idx = np.arange(seg.shape[1], dtype=np.float32).reshape(1, -1, 1)
    mask_ilm = seg == LAYER_ILM
    mask_rpe = seg == LAYER_RPE
    count_ilm = mask_ilm.sum(axis=1)
    count_rpe = mask_rpe.sum(axis=1)
    sum_ilm = (mask_ilm * depth_idx).sum(axis=1)
    sum_rpe = (mask_rpe * depth_idx).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ilm = sum_ilm / np.maximum(count_ilm, 1)
        mean_rpe = sum_rpe / np.maximum(count_rpe, 1)
    valid = (count_ilm > 0) & (count_rpe > 0)
    thickness = np.where(valid, mean_rpe - mean_ilm, np.nan).astype(np.float32)
    return thickness, valid, count_ilm, count_rpe


def block_nanmean(arr: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    h, w = arr.shape
    bh, bw = h // grid_h, w // grid_w
    reshaped = arr[:grid_h * bh, :grid_w * bw].reshape(grid_h, bh, grid_w, bw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(reshaped, axis=(1, 3))


def extract_case_features(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Devuelve (compact_features, grid_features, diagnostics)."""
    thickness, valid, count_ilm, count_rpe = compute_thickness_map(seg)
    uu, vv = _normalized_coords(thickness.shape)

    valid_thickness = thickness[valid]
    if valid_thickness.size == 0:
        thickness_mean = thickness_std = thickness_min = thickness_max = 0.0
        thickness_p10 = thickness_p90 = 0.0
        argmin_u = argmin_v = argmax_u = argmax_v = -1.0
    else:
        thickness_mean = float(valid_thickness.mean())
        thickness_std = float(valid_thickness.std())
        thickness_min = float(valid_thickness.min())
        thickness_max = float(valid_thickness.max())
        thickness_p10 = float(np.percentile(valid_thickness, 10))
        thickness_p90 = float(np.percentile(valid_thickness, 90))
        masked = np.where(valid, thickness, np.nan)
        idx_min = int(np.nanargmin(masked))
        idx_max = int(np.nanargmax(masked))
        argmin_v, argmin_u = np.unravel_index(idx_min, masked.shape)
        argmax_v, argmax_u = np.unravel_index(idx_max, masked.shape)
        argmin_u, argmin_v = float(uu[argmin_v, argmin_u]), float(vv[argmin_v, argmin_u])
        argmax_u, argmax_v = float(uu[argmax_v, argmax_u]), float(vv[argmax_v, argmax_u])

    dip_weight = np.where(valid, np.clip(thickness_mean - thickness, 0.0, None), 0.0)
    dip_centroid_u, dip_centroid_v, dip_weight_total = _weighted_centroid(dip_weight, uu, vv)

    missing_map = (~valid).astype(np.float32)
    missing_centroid_u, missing_centroid_v, _ = _weighted_centroid(missing_map, uu, vv)
    missing_fraction = float(missing_map.mean())

    thickness_filled = np.where(valid, thickness, thickness_mean)
    gy, gx = np.gradient(thickness_filled)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    thickness_grad_mean = float(grad_mag[valid].mean()) if valid.any() else 0.0
    thickness_grad_std = float(grad_mag[valid].std()) if valid.any() else 0.0

    vessel_density = (seg == VESSEL_CLASS).astype(np.float32).mean(axis=1)
    vessel_density_mean = float(vessel_density.mean())
    vessel_density_std = float(vessel_density.std())
    vessel_centroid_u, vessel_centroid_v, _ = _weighted_centroid(vessel_density, uu, vv)

    instrument_density = np.isin(seg, INSTRUMENT_CLASSES).astype(np.float32).mean(axis=1)
    instrument_present = float(instrument_density.sum() > 0)
    instrument_density_mean = float(instrument_density.mean())
    instrument_centroid_u, instrument_centroid_v, _ = _weighted_centroid(instrument_density, uu, vv)

    uncaptured = (seg == UNCAPTURED_CLASS).astype(np.float32).mean(axis=1)
    uncaptured_fraction = float(uncaptured.mean())
    uncaptured_centroid_u, uncaptured_centroid_v, _ = _weighted_centroid(uncaptured, uu, vv)

    compact = np.array([
        thickness_mean, thickness_std, thickness_min, thickness_max,
        thickness_p10, thickness_p90, thickness_max - thickness_min, float(valid.mean()),
        argmin_u, argmin_v, argmax_u, argmax_v,
        dip_centroid_u, dip_centroid_v, dip_weight_total,
        missing_centroid_u, missing_centroid_v, missing_fraction,
        thickness_grad_mean, thickness_grad_std,
        vessel_density_mean, vessel_density_std, vessel_centroid_u, vessel_centroid_v,
        instrument_present, instrument_density_mean, instrument_centroid_u, instrument_centroid_v,
        uncaptured_fraction, uncaptured_centroid_u, uncaptured_centroid_v,
        float(count_ilm.sum()), float(count_rpe.sum()),
    ], dtype=np.float64)
    assert compact.shape[0] == len(COMPACT_FEATURE_NAMES)

    thickness_grid = block_nanmean(thickness_filled, GRID, GRID)
    vessel_grid = block_nanmean(vessel_density, GRID, GRID)
    instrument_grid = block_nanmean(instrument_density, GRID, GRID)
    grid = np.concatenate([thickness_grid.ravel(), vessel_grid.ravel(), instrument_grid.ravel()]).astype(np.float64)

    diagnostics = {
        "thickness_mean": thickness_mean, "thickness_std": thickness_std,
        "thickness_range": thickness_max - thickness_min,
        "valid_fraction": float(valid.mean()),
        "dip_weight_total": dip_weight_total,
        "instrument_present": instrument_present,
        "uncaptured_fraction": uncaptured_fraction,
    }
    return compact, grid, diagnostics


# --------------------------------------------------------------------------
# 4. Extraccion del dataset completo (features + targets + grupos)
# --------------------------------------------------------------------------

def build_dataset(cases: list[dict], data_root: Path, log_every: int = 100) -> dict:
    reader = ScenarioVolumeReader(data_root)
    n = len(cases)
    X_compact = np.zeros((n, len(COMPACT_FEATURE_NAMES)), dtype=np.float64)
    X_grid = np.zeros((n, 3 * GRID * GRID), dtype=np.float64)
    tx = np.zeros(n); ty = np.zeros(n)
    cos_theta = np.zeros(n); sin_theta = np.zeros(n); scale = np.zeros(n)
    gt_matrices = np.zeros((n, 3, 3), dtype=np.float64)
    scenarios = []
    case_ids = []
    diagnostics_rows = []
    kept = np.zeros(n, dtype=bool)
    warnings_log = []

    started = time.monotonic()
    for i, case in enumerate(cases):
        case_id = f"{case['scenario']}/{case['frame_id']}"
        try:
            data = json.loads(case["json_path"].read_text(encoding="utf-8"))
            matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
            params = decompose_similarity(matrix, reflect=True)
            seg = reader.read_segmentation(case["scenario"], case["frame_id"])
            compact, grid, diag = extract_case_features(seg)
        except Exception as exc:  # noqa: BLE001 - se reporta y se sigue
            message = f"{case_id}: omitido, {type(exc).__name__}: {exc}"
            warnings.warn(message)
            warnings_log.append(message)
            continue

        X_compact[i] = compact
        X_grid[i] = grid
        tx[i] = float(params["tx"]); ty[i] = float(params["ty"])
        cos_theta[i] = float(params["cos_theta"]); sin_theta[i] = float(params["sin_theta"])
        scale[i] = float(params["scale"])
        gt_matrices[i] = matrix
        scenarios.append(case["scenario"])
        case_ids.append(case_id)
        diag["case_id"] = case_id
        diagnostics_rows.append(diag)
        kept[i] = True

        if (i + 1) % log_every == 0:
            elapsed = time.monotonic() - started
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else float("nan")
            print(f"[{i + 1}/{n}] {rate:.1f} casos/min", flush=True)

    reader.close()
    scenarios = np.asarray(scenarios, dtype=str)
    case_ids_arr = np.asarray(case_ids, dtype=str)

    return {
        "X_compact": X_compact[kept], "X_grid": X_grid[kept],
        "tx": tx[kept], "ty": ty[kept],
        "cos_theta": cos_theta[kept], "sin_theta": sin_theta[kept], "scale": scale[kept],
        "gt_matrices": gt_matrices[kept], "scenarios": scenarios, "case_ids": case_ids_arr,
        "diagnostics_rows": diagnostics_rows, "n_requested": n, "n_kept": int(kept.sum()),
        "warnings": warnings_log,
    }


# --------------------------------------------------------------------------
# 5. Modelado: GroupKFold leave-one-scenario-out + control shuffle
# --------------------------------------------------------------------------

def make_models(seed: int) -> dict:
    return {
        "ridge": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, early_stopping=True,
                        random_state=seed, alpha=1e-2)),
    }


def group_kfold_splits(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X=np.zeros((len(groups), 1)), groups=groups))


def oof_predict(X: np.ndarray, Y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
                model_fn) -> np.ndarray:
    oof = np.full_like(np.asarray(Y, dtype=np.float64), np.nan)
    for train_idx, test_idx in splits:
        model = model_fn()
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        oof[test_idx] = pred
    return oof


def fold_constants(cos_train: np.ndarray, sin_train: np.ndarray, scale_train: np.ndarray) -> tuple[float, float, float]:
    """Media circular de theta (evita el problema de wraparound en +-180deg,
    confirmado real en estos datos) + mediana de la escala."""
    vec = np.array([np.mean(cos_train), np.mean(sin_train)])
    norm = np.linalg.norm(vec)
    cos_c, sin_c = (vec / norm) if norm > 1e-8 else (1.0, 0.0)
    return float(cos_c), float(sin_c), float(np.median(scale_train))


def matrices_from_position_and_fold_constants(tx_pred, ty_pred, cos_theta, sin_theta, scale,
                                              splits) -> np.ndarray:
    n = len(tx_pred)
    matrices = np.zeros((n, 3, 3), dtype=np.float64)
    for train_idx, test_idx in splits:
        cos_c, sin_c, scale_c = fold_constants(cos_theta[train_idx], sin_theta[train_idx], scale[train_idx])
        for i in test_idx:
            matrices[i] = np.asarray(
                compose_similarity(tx_pred[i], ty_pred[i], cos_c, sin_c, scale_c, reflect=True))
    return matrices


def matrices_from_full_prediction(tx_pred, ty_pred, cos_pred, sin_pred, scale_pred) -> np.ndarray:
    n = len(tx_pred)
    matrices = np.zeros((n, 3, 3), dtype=np.float64)
    for i in range(n):
        norm = np.hypot(cos_pred[i], sin_pred[i])
        cos_i, sin_i = (cos_pred[i] / norm, sin_pred[i] / norm) if norm > 1e-8 else (1.0, 0.0)
        matrices[i] = np.asarray(
            compose_similarity(tx_pred[i], ty_pred[i], cos_i, sin_i, scale_pred[i], reflect=True))
    return matrices


def circular_r2(cos_true, sin_true, cos_pred, sin_pred) -> float:
    """R^2 conjunto de (cos,sin) -- 1 - SS_res/SS_tot en el plano unitario."""
    ss_res = np.sum((cos_true - cos_pred) ** 2 + (sin_true - sin_pred) ** 2)
    ss_tot = np.sum((cos_true - cos_true.mean()) ** 2 + (sin_true - sin_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mean_angular_error_deg(cos_true, sin_true, cos_pred, sin_pred) -> float:
    theta_true = np.arctan2(sin_true, cos_true)
    theta_pred = np.arctan2(sin_pred, cos_pred)
    diff = np.abs(np.degrees(np.angle(np.exp(1j * (theta_true - theta_pred)))))
    return float(np.mean(diff))


def name_index(name: str) -> int:
    return COMPACT_FEATURE_NAMES.index(name)


# --------------------------------------------------------------------------
# Cache de dataset (para no releer los zips en cada corrida de modelado)
# --------------------------------------------------------------------------

def load_or_build_dataset(cases: list[dict], data_root: Path, cache_path: Path,
                          refresh: bool, log_every: int) -> dict:
    if cache_path.exists() and not refresh:
        print(f"Cargando cache de features desde {cache_path}", flush=True)
        payload = np.load(cache_path, allow_pickle=False)
        return {
            "X_compact": payload["X_compact"], "X_grid": payload["X_grid"],
            "tx": payload["tx"], "ty": payload["ty"],
            "cos_theta": payload["cos_theta"], "sin_theta": payload["sin_theta"],
            "scale": payload["scale"], "gt_matrices": payload["gt_matrices"],
            "scenarios": payload["scenarios"].astype(str), "case_ids": payload["case_ids"].astype(str),
            "n_requested": int(payload["n_requested"]), "n_kept": int(payload["n_kept"]),
            "warnings": [],
        }

    print(f"Extrayendo features de {len(cases)} casos (root={data_root})...", flush=True)
    dataset = build_dataset(cases, data_root, log_every=log_every)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, X_compact=dataset["X_compact"], X_grid=dataset["X_grid"],
        tx=dataset["tx"], ty=dataset["ty"], cos_theta=dataset["cos_theta"],
        sin_theta=dataset["sin_theta"], scale=dataset["scale"],
        gt_matrices=dataset["gt_matrices"], scenarios=dataset["scenarios"],
        case_ids=dataset["case_ids"], n_requested=dataset["n_requested"], n_kept=dataset["n_kept"],
    )
    if dataset["warnings"]:
        warnings_path = cache_path.with_suffix(".warnings.txt")
        warnings_path.write_text("\n".join(dataset["warnings"]) + "\n", encoding="utf-8")
        print(f"{len(dataset['warnings'])} casos omitidos; detalle en {warnings_path}", flush=True)
    print(f"Features cacheadas en {cache_path} ({dataset['n_kept']}/{dataset['n_requested']} casos)", flush=True)
    return dataset


# --------------------------------------------------------------------------
# 6. Pipeline completo de modelado
# --------------------------------------------------------------------------

def run_pipeline(dataset: dict, seed: int, n_splits: int = N_SPLITS) -> dict:
    require_task2_training_root(REPO_ROOT / "data" / "Task 2")  # rechaza rutas Mock Test

    X_compact = dataset["X_compact"]
    X_full = np.concatenate([dataset["X_compact"], dataset["X_grid"]], axis=1)
    tx, ty = dataset["tx"], dataset["ty"]
    cos_theta, sin_theta, scale = dataset["cos_theta"], dataset["sin_theta"], dataset["scale"]
    gt_matrices = dataset["gt_matrices"]
    scenarios, case_ids = dataset["scenarios"], dataset["case_ids"]
    n = len(tx)

    splits = group_kfold_splits(scenarios, n_splits)
    feature_sets = {"compact": X_compact, "full": X_full}
    models = make_models(seed)

    position_target = np.stack([tx, ty], axis=1)
    rotation_target = np.stack([cos_theta, sin_theta], axis=1)

    results: dict = {"models": {}, "n_cases": n, "n_scenarios": len(set(scenarios.tolist()))}

    # --- Baseline: pose constante (train-median de las 4 componentes) ---
    const_matrices = np.zeros_like(gt_matrices)
    for train_idx, test_idx in splits:
        cos_c, sin_c, scale_c = fold_constants(cos_theta[train_idx], sin_theta[train_idx], scale[train_idx])
        tx_c, ty_c = float(np.median(tx[train_idx])), float(np.median(ty[train_idx]))
        for i in test_idx:
            const_matrices[i] = np.asarray(compose_similarity(tx_c, ty_c, cos_c, sin_c, scale_c, reflect=True))
    const_eval = evaluate_task2_cases(const_matrices, gt_matrices, scenarios, case_ids=case_ids)
    results["constant_pose_baseline"] = {
        "auc": const_eval["auc"], "mean_error": const_eval["mean_error"],
        "median_error": const_eval["median_error"],
    }
    print(f"[control] pose constante (train-median, sin OCT): AUC={const_eval['auc']:.4f} "
          f"mean_error={const_eval['mean_error']:.2f}px", flush=True)

    best_key = None
    best_auc = -1.0
    for fs_name, X in feature_sets.items():
        for model_name, model_fn in models.items():
            key = f"{fs_name}/{model_name}"
            t0 = time.monotonic()
            oof_pos = oof_predict(X, position_target, splits, model_fn)
            oof_rot = oof_predict(X, rotation_target, splits, model_fn)
            oof_scale = oof_predict(X, scale, splits, model_fn)
            elapsed = time.monotonic() - t0

            r2_tx = r2_score(tx, oof_pos[:, 0])
            r2_ty = r2_score(ty, oof_pos[:, 1])
            r2_pos = r2_score(position_target, oof_pos, multioutput="uniform_average")
            r2_theta = circular_r2(cos_theta, sin_theta, oof_rot[:, 0], oof_rot[:, 1])
            angle_err = mean_angular_error_deg(cos_theta, sin_theta, oof_rot[:, 0], oof_rot[:, 1])
            r2_scale = r2_score(scale, oof_scale)

            pred_matrices_median_rot_scale = matrices_from_position_and_fold_constants(
                oof_pos[:, 0], oof_pos[:, 1], cos_theta, sin_theta, scale, splits)
            eval_pos_only = evaluate_task2_cases(pred_matrices_median_rot_scale, gt_matrices,
                                                 scenarios, case_ids=case_ids)

            pred_matrices_full = matrices_from_full_prediction(
                oof_pos[:, 0], oof_pos[:, 1], oof_rot[:, 0], oof_rot[:, 1], oof_scale)
            eval_full = evaluate_task2_cases(pred_matrices_full, gt_matrices, scenarios, case_ids=case_ids)

            entry = {
                "r2_tx": float(r2_tx), "r2_ty": float(r2_ty), "r2_position": float(r2_pos),
                "r2_theta_circular": float(r2_theta), "mean_angular_error_deg": float(angle_err),
                "r2_scale": float(r2_scale),
                "auc_position_plus_median_rotation_scale": eval_pos_only["auc"],
                "mean_error_position_plus_median_rotation_scale": eval_pos_only["mean_error"],
                "auc_full_prediction": eval_full["auc"],
                "mean_error_full_prediction": eval_full["mean_error"],
                "per_scenario_auc_position_plus_median": {
                    s: v["auc"] for s, v in eval_pos_only["per_scenario"].items()},
                "fit_seconds": elapsed,
                "n_features": X.shape[1],
            }
            results["models"][key] = entry
            print(f"[{key}] R2(tx,ty)={r2_pos:.4f} R2(theta)={r2_theta:.4f} R2(scale)={r2_scale:.4f} "
                  f"AUC(pos+mediana)={eval_pos_only['auc']:.4f} AUC(todo predicho)={eval_full['auc']:.4f} "
                  f"[{elapsed:.1f}s]", flush=True)

            if eval_pos_only["auc"] > best_auc:
                best_auc = eval_pos_only["auc"]
                best_key = key

    results["best_model"] = best_key
    results["best_auc"] = best_auc

    # --- Control: barajar las features del OCT entre casos ---
    fs_name, model_name = best_key.split("/")
    X_best = feature_sets[fs_name]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    X_shuffled = X_best[permutation]

    model_fn = models[model_name]
    oof_pos_shuffled = oof_predict(X_shuffled, position_target, splits, model_fn)
    r2_pos_shuffled = r2_score(position_target, oof_pos_shuffled, multioutput="uniform_average")
    pred_matrices_shuffled = matrices_from_position_and_fold_constants(
        oof_pos_shuffled[:, 0], oof_pos_shuffled[:, 1], cos_theta, sin_theta, scale, splits)
    eval_shuffled = evaluate_task2_cases(pred_matrices_shuffled, gt_matrices, scenarios, case_ids=case_ids)
    results["shuffle_control"] = {
        "feature_set": fs_name, "model": model_name,
        "r2_position": float(r2_pos_shuffled), "auc": eval_shuffled["auc"],
        "mean_error": eval_shuffled["mean_error"],
    }
    print(f"[control shuffle] {fs_name}/{model_name} con features barajadas: "
          f"R2(tx,ty)={r2_pos_shuffled:.4f} AUC={eval_shuffled['auc']:.4f}", flush=True)

    # --- Premisa anatomica: distribucion de variacion de espesor Ilm-Rpe ---
    thickness_std = X_compact[:, name_index("thickness_std")]
    thickness_range = X_compact[:, name_index("thickness_range")]
    valid_fraction = X_compact[:, name_index("valid_fraction")]
    dip_weight_total = X_compact[:, name_index("dip_weight_total")]
    thickness_mean = X_compact[:, name_index("thickness_mean")]
    flat_threshold_px = 5.0
    results["anatomical_premise"] = {
        "thickness_std_mean": float(thickness_std.mean()), "thickness_std_median": float(np.median(thickness_std)),
        "thickness_range_mean": float(thickness_range.mean()), "thickness_range_median": float(np.median(thickness_range)),
        "thickness_mean_mean": float(thickness_mean.mean()),
        "valid_fraction_mean": float(valid_fraction.mean()), "valid_fraction_min": float(valid_fraction.min()),
        "dip_weight_total_mean": float(dip_weight_total.mean()),
        "flat_case_fraction": float(np.mean(thickness_range < flat_threshold_px)),
        "flat_threshold_px": flat_threshold_px,
        "instrument_present_fraction": float(X_compact[:, name_index("instrument_present")].mean()),
        "uncaptured_fraction_mean": float(X_compact[:, name_index("uncaptured_fraction")].mean()),
    }

    return results


# --------------------------------------------------------------------------
# 7. Informe
# --------------------------------------------------------------------------

def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(results: dict, dataset: dict, output_path: Path, args: argparse.Namespace) -> None:
    n = results["n_cases"]
    premise = results["anatomical_premise"]
    const = results["constant_pose_baseline"]
    best_key = results["best_model"]
    best = results["models"][best_key]
    shuffle = results["shuffle_control"]

    lines: list[str] = []
    lines += [
        "# Auto-localizacion del OCT en Task 2 -- el volumen, sin mirar el fundus",
        "",
        f"Casos usados: **{n}** de {dataset['n_requested']} solicitados "
        f"({results['n_scenarios']} escenarios). GroupKFold leave-one-scenario-out, "
        f"n_splits={N_SPLITS}. Semilla: {args.seed}.",
        "",
        "Pregunta: ¿el volumen OCT, por si solo, codifica donde esta situado en la "
        "retina? Ningun feature de este script toca `microscope.png` (el fundus) en "
        "ningun punto -- todas las features vienen de `iOCT Microscope/Volume/.../Segmentation`.",
        "",
        "## 0. Premisa anatomica (paso 3 del pedido)",
        "",
        f"Clases de segmentacion disponibles en el volumen (`vendor/fido/Dataset Explorer/constants.py`, "
        "`GenericLabels`): `Ilm=1`, `Rpe=2` (unicas superficies de capa retinal), mas "
        "`ArteriesOrVeins=3`, instrumento (`Forceps=8`, `Endoilluminator=10`, `InstrumentInOCT=11`), "
        "`OCTUncapturedArea=14`. **No existe una clase dedicada a fovea ni a disco optico** -- si hay "
        "senal anatomica, tiene que verse en la FORMA del mapa de espesor Ilm-Rpe, no en una etiqueta.",
        "",
        f"Espesor Ilm-Rpe por columna A-scan, agregado sobre los {n} casos: media de la media por caso "
        f"**{fmt(premise['thickness_mean_mean'], 1)} px**; desviacion estandar DENTRO de cada volumen "
        f"(media sobre casos) **{fmt(premise['thickness_std_mean'], 1)} px**; rango max-min dentro de "
        f"cada volumen (media sobre casos) **{fmt(premise['thickness_range_mean'], 1)} px**.",
        "",
        f"Fraccion de casos 'planos' (rango de espesor < {premise['flat_threshold_px']:.0f} px, "
        f"es decir sin variacion apreciable): **{fmt(premise['flat_case_fraction'])}**.",
        "",
        f"Cobertura valida (columnas con Ilm y Rpe detectados) media: **{fmt(premise['valid_fraction_mean'])}** "
        f"(minimo observado: {fmt(premise['valid_fraction_min'])}). Fraccion de columnas 'OCTUncapturedArea': "
        f"**{fmt(premise['uncaptured_fraction_mean'], 4)}**. Casos con instrumento visible en el volumen: "
        f"**{fmt(premise['instrument_present_fraction'])}**.",
        "",
    ]

    if premise["flat_case_fraction"] > 0.8:
        lines.append(
            "**Veredicto de premisa**: la gran mayoria de los volumenes son esencialmente planos -- "
            "el simulador NO modela variacion anatomica de espesor retinal. La hipotesis muere aqui "
            "para la via de espesor de capas (ver igualmente la seccion 1 para el resto de features).")
    else:
        lines.append(
            f"**Veredicto de premisa**: el simulador SI produce variacion de espesor no trivial dentro "
            f"de cada volumen (std media {fmt(premise['thickness_std_mean'],1)} px sobre una media de "
            f"{fmt(premise['thickness_mean_mean'],1)} px, es decir "
            f"~{fmt(100*premise['thickness_std_mean']/max(premise['thickness_mean_mean'],1e-6),1)}% de "
            "variacion relativa). La premisa anatomica NO esta descartada de entrada; el peso de la "
            "prueba pasa a la seccion 1: si esa variacion esta ligada a la posicion real (tx,ty) o si "
            "es una textura sin relacion con la pose (lo que decide el control de barajado).")
    lines.append("")

    lines += [
        "## 1. Test central: (tx, ty) solo con OCT",
        "",
        f"Control (0 features de OCT, pose constante = mediana de train por fold): "
        f"AUC=**{fmt(const['auc'])}**, error medio={fmt(const['mean_error'],2)} px. "
        "(Reproduce el hallazgo previo de pose constante; sirve de verificacion de metodologia.)",
        "",
        "| feature set / modelo | R2(tx,ty) | R2(tx) | R2(ty) | AUC pos+mediana(theta,s) | AUC todo-predicho | tiempo |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in sorted(results["models"].items(), key=lambda kv: -kv[1]["auc_position_plus_median_rotation_scale"]):
        marker = " **<- mejor**" if key == best_key else ""
        lines.append(
            f"| {key}{marker} | {fmt(entry['r2_position'])} | {fmt(entry['r2_tx'])} | {fmt(entry['r2_ty'])} | "
            f"{fmt(entry['auc_position_plus_median_rotation_scale'])} | {fmt(entry['auc_full_prediction'])} | "
            f"{entry['fit_seconds']:.1f}s |")
    lines += [
        "",
        f"Mejor combinacion: **{best_key}**, AUC(posicion OCT + theta/escala mediana de train)="
        f"**{fmt(best['auc_position_plus_median_rotation_scale'])}** "
        f"(vs. {fmt(const['auc'])} de la pose constante, vs. 0.475 del lider del leaderboard).",
        "",
        "AUC por escenario retenido (mejor modelo, posicion OCT + mediana theta/escala):",
        "",
        "| escenario | AUC |", "|---|---:|",
    ]
    for scenario, auc in sorted(best["per_scenario_auc_position_plus_median"].items()):
        lines.append(f"| {scenario} | {fmt(auc)} |")

    lines += [
        "",
        "## 2. Rotacion y escala por separado",
        "",
        "| feature set / modelo | R2(theta, circular) | error angular medio (deg) | R2(escala) |",
        "|---|---:|---:|---:|",
    ]
    for key, entry in sorted(results["models"].items()):
        lines.append(f"| {key} | {fmt(entry['r2_theta_circular'])} | {fmt(entry['mean_angular_error_deg'],1)} | "
                     f"{fmt(entry['r2_scale'])} |")

    lines += [
        "",
        "## 3. Control obligatorio -- OCT barajado entre casos",
        "",
        f"Mismo modelo y feature set ganador (**{shuffle['feature_set']}/{shuffle['model']}**), features del "
        "OCT permutadas entre casos con semilla fija (targets y grupos SIN barajar):",
        "",
        f"- R2(tx,ty) con features barajadas: **{fmt(shuffle['r2_position'])}** "
        f"(vs. {fmt(best['r2_position'])} sin barajar)",
        f"- AUC con features barajadas: **{fmt(shuffle['auc'])}** "
        f"(vs. {fmt(best['auc_position_plus_median_rotation_scale'])} sin barajar, "
        f"vs. {fmt(const['auc'])} de la pose constante)",
        "",
    ]
    if shuffle["auc"] <= const["auc"] + 0.01 and shuffle["r2_position"] <= 0.02:
        lines.append("El control colapsa a nivel del piso constante: la senal medida en la seccion 1 "
                     "no es un artefacto de indexado/orden -- depende genuinamente de que features y "
                     "targets esten emparejados por caso.")
    else:
        lines.append("**ALERTA**: el control de barajado NO colapsa del todo -- parte de la senal "
                     "reportada en la seccion 1 podria ser un artefacto (p.ej. estructura de escenario "
                     "que sobrevive incluso sin la correspondencia caso a caso). Revisar antes de sacar "
                     "conclusiones fuertes.")

    lines += [
        "",
        "## 4. Interpretacion pre-registrada",
        "",
    ]
    auc_best = best["auc_position_plus_median_rotation_scale"]
    if auc_best >= 0.10:
        lines.append(
            f"**AUC={fmt(auc_best)} >= 0.10: hipotesis FUERTE.** Hay un frente nuevo real: el volumen OCT, "
            "sin ver el fundus, aporta senal de localizacion suficiente para mover la aguja del score.")
    elif best["r2_position"] > 0.05:
        lines.append(
            f"**R2(tx,ty)={fmt(best['r2_position'])} apreciable pero AUC={fmt(auc_best)} < 0.10.** Hay senal "
            "real (sobrevive el control de barajado) pero insuficiente por si sola para competir con el "
            "estado actual del leaderboard. Util como prior/feature auxiliar dentro de un modelo con fundus, "
            "no como solucion standalone.")
    else:
        lines.append(
            f"**R2(tx,ty)={fmt(best['r2_position'])} ~ 0.** El volumen OCT, por si solo, no predice donde "
            "esta situado en la retina con este set de features. La hipotesis de auto-localizacion muere "
            "para el enfoque probado aqui.")

    lines += [
        "",
        "## Notas de implementacion",
        "",
        "- Metrica oficial: `src/fido/eval_task2.py::evaluate_task2_cases` (misma funcion que otros "
        "diagnosticos de Task 2 en este repo), que envuelve `corner_auc`/`corner_error` de "
        "`src/fido/geometry.py`, replica exacta del scorer oficial vendorizado.",
        f"- GroupKFold con {N_SPLITS} folds sobre {results['n_scenarios']} escenarios: cada fold deja "
        "fuera un escenario completo (leave-one-scenario-out exacto, no aproximado).",
        "- theta/escala 'mediana de train' se calculan POR FOLD sobre los escenarios de entrenamiento de "
        "ese fold (media circular para theta -- el angulo cruza +-180 grados en los datos reales, "
        "confirmado; mediana simple para la escala).",
        f"- Features compactas: {len(COMPACT_FEATURE_NAMES)} (interpretables: estadisticas de espesor, "
        "posicion del minimo/maximo, centroide de la 'zona fina', densidad y centroide de vasos e "
        f"instrumento). Features 'full': compactas + mapas {GRID}x{GRID} de espesor/vasos/instrumento "
        f"({3 * GRID * GRID} valores adicionales).",
        f"- Cache de features en `{args.cache_path}`; borrar o pasar `--refresh` para reextraer desde los "
        "zips/directorios de `data/Task 2`.",
    ]
    if dataset.get("warnings"):
        lines += ["", f"- {len(dataset['warnings'])} casos omitidos por error de carga; detalle en "
                 f"`{args.cache_path.with_suffix('.warnings.txt')}`."]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe escrito en {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, default=REPO_ROOT / "data/_annotations/Task 2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/Task 2")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/96-t2-oct-self-localisation")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--per-scenario-limit", type=int, default=None,
                        help="recorta cada escenario a N casos (smoke test); preserva los 10 grupos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="reextraer features aunque exista cache")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.cache_path is None:
        args.cache_path = args.output_dir / "features_cache.npz"

    cases = discover_cases(args.annotations_root, per_scenario_limit=args.per_scenario_limit)
    if not cases:
        raise SystemExit(f"no se encontraron casos en {args.annotations_root}")
    print(f"{len(cases)} casos descubiertos en {args.annotations_root}", flush=True)

    dataset = load_or_build_dataset(cases, args.data_root, args.cache_path, args.refresh,
                                    args.log_every)
    print(f"Dataset final: {dataset['n_kept']} casos usables", flush=True)

    results = run_pipeline(dataset, seed=args.seed)

    results_path = args.output_dir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"Resultados crudos en {results_path}", flush=True)

    write_report(results, dataset, args.output_dir / "INFORME.md", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
