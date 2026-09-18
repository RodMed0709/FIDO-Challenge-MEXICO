"""Task 2 como estimacion de pose (reformulacion T2-98), no registracion de imagenes.

Hallazgo del coordinador (`experiments/98-t2-pose-estimation/HALLAZGO.md`,
verificado de forma independiente en este script -- ver `verify_hallazgo()`):
sobre los 1214 casos de train, `Opmi.Spatial` es constante, `Eyeball.Translation`
es constante (cero), y solo dos cuaterniones varian: `iOCT Microscope.Spatial.Rotation`
(rotacion del escaner) y `Eyeball.Spatial.Rotation` (rotacion del ojo). Una
regresion cuadratica de esos 8 componentes contra los 6 parametros de la matriz
GT da R2 0.975-0.994 y error absoluto medio 5.7-7.0 px en los terminos de
traslacion (dentro de la ventana 0-10px que puntua la metrica oficial).

Este script responde las 3 preguntas del pedido revisado:

1. ¿Se puede estimar `iOCT.Rotation` (4 componentes) desde el VOLUMEN OCT?
   Reusa las features ya extraidas por `measure_task2_oct_self_localisation.py`
   (cache en `experiments/96-t2-oct-self-localisation/features_cache.npz`) --
   el trabajo de esa corrida no se tira, solo cambia el target.
2. ¿Se puede estimar `Eyeball.Rotation` (4 componentes) desde el FUNDUS?
   Features nuevas: mascara de retina visible (ilm.png), mascara de vasos
   (arteriesorveins.png), y brillo (proxy del disco optico, mas brillante que
   el resto de la retina).
3. ¿Que precision angular hace falta para bajar de 10px de error de esquinas?
   Se perturbán los cuaterniones GT con ruido angular creciente, se pasan por
   una regresion cuadratica sustituta (cuaterniones -> 6 parametros de matriz,
   ajustada sobre los 1214 casos como aproximacion de la geometria fija del
   simulador) y se mide el corner-AUC oficial resultante.

GroupKFold por escenario (leave-one-scenario-out, n_splits=10) y control de
barajado obligatorios en los pasos 1 y 2, igual que en el resto de la escalera.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from measure_task2_oct_self_localisation import (  # noqa: E402
    GRID, N_SPLITS, _normalized_coords, _weighted_centroid, block_nanmean,
    group_kfold_splits, make_models, oof_predict,
)
from fido.eval_task2 import evaluate_task2_cases, require_task2_training_root  # noqa: E402
from fido.geometry import compose_similarity, corner_auc, corner_error  # noqa: E402

VESSEL_THRESHOLD = 0  # arteriesorveins.png / ilm.png son mascaras binarias 0/255
BRIGHT_PERCENTILE = 95.0

FUNDUS_COMPACT_FEATURE_NAMES = [
    "ilm_area_fraction", "ilm_centroid_u", "ilm_centroid_v",
    "ilm_orientation_cos2", "ilm_orientation_sin2", "ilm_elongation",
    "vessel_density", "vessel_centroid_u", "vessel_centroid_v",
    "vessel_orientation_cos2", "vessel_orientation_sin2", "vessel_elongation",
    "bright_centroid_u", "bright_centroid_v",
    "mean_intensity_in_ilm", "std_intensity_in_ilm",
]


# --------------------------------------------------------------------------
# 0. Verificacion independiente del hallazgo T2-98
# --------------------------------------------------------------------------

def verify_hallazgo(annotations_root: Path) -> dict:
    files = sorted(annotations_root.glob("Scenario_*/*.json"))
    opmi_r, ioct_r, eye_r, eye_t = [], [], [], []
    mat6 = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        opmi_r.append(data["Opmi"]["Spatial"]["Rotation"])
        ioct_r.append(data["iOCT Microscope"]["Spatial"]["Rotation"])
        eye_r.append(data["Eyeball"]["Spatial"]["Rotation"])
        eye_t.append(data["Eyeball"]["Spatial"]["Translation"])
        m = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
        mat6.append([m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2]])
    opmi_r, ioct_r, eye_r, eye_t = (np.asarray(a, dtype=np.float64) for a in (opmi_r, ioct_r, eye_r, eye_t))
    mat6 = np.asarray(mat6, dtype=np.float64)

    X = np.concatenate([ioct_r, eye_r], axis=1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    surrogate = LinearRegression().fit(Xp, mat6)
    pred = surrogate.predict(Xp)
    r2_per_param = [float(r2_score(mat6[:, i], pred[:, i])) for i in range(6)]
    mae_per_param = np.abs(mat6 - pred).mean(axis=0).tolist()

    return {
        "n_cases": len(files),
        "opmi_rotation_std": opmi_r.std(axis=0).tolist(),
        "ioct_rotation_std": ioct_r.std(axis=0).tolist(),
        "eyeball_rotation_std": eye_r.std(axis=0).tolist(),
        "eyeball_translation_std": eye_t.std(axis=0).tolist(),
        "surrogate_quadratic_r2_per_matrix_param": r2_per_param,
        "surrogate_quadratic_mae_per_matrix_param": mae_per_param,
        "poly": poly, "surrogate_model": surrogate,
    }


# --------------------------------------------------------------------------
# 1. Reader de fundus (mismo patron que ScenarioVolumeReader del script OCT)
# --------------------------------------------------------------------------

class FundusReader:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._zip_cache: dict[str, zipfile.ZipFile] = {}

    def _zip(self, scenario: str) -> zipfile.ZipFile:
        if scenario not in self._zip_cache:
            self._zip_cache[scenario] = zipfile.ZipFile(self.data_root / f"{scenario}.zip")
        return self._zip_cache[scenario]

    def _open(self, scenario: str, relpath: str) -> Image.Image:
        extracted = self.data_root / scenario / relpath
        if extracted.is_file():
            return Image.open(extracted)
        zf = self._zip(scenario)
        return Image.open(io.BytesIO(zf.read(relpath)))

    def read_case(self, scenario: str, frame_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (gray uint8 1024x1024, ilm_mask bool, vessel_mask bool)."""
        gray = np.array(self._open(scenario, f"Stereo Left/{frame_id}/microscope.png").convert("L"))
        ilm = np.array(self._open(scenario, f"Stereo Left/{frame_id}/Segmentation/ilm.png")) > VESSEL_THRESHOLD
        vessel = np.array(self._open(
            scenario, f"Stereo Left/{frame_id}/Segmentation/arteriesorveins.png")) > VESSEL_THRESHOLD
        return gray, ilm, vessel

    def close(self) -> None:
        for zf in self._zip_cache.values():
            zf.close()


# --------------------------------------------------------------------------
# 2. Features de fundus -- SOLO del lado microscopio, nunca del OCT
# --------------------------------------------------------------------------

def _orientation_features(mask: np.ndarray, uu: np.ndarray, vv: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Centroide + orientacion principal (doble angulo, evita la ambiguedad de
    180 grados de un eje) + elongacion, todo ponderado por `mask`."""
    weight = mask.astype(np.float32)
    cu, cv, total = _weighted_centroid(weight, uu, vv)
    if total <= 1e-6:
        return cu, cv, -1.0, -1.0, 0.0, 0.0
    du = uu - cu
    dv = vv - cv
    sxx = float(np.sum(weight * du * du) / total)
    syy = float(np.sum(weight * dv * dv) / total)
    sxy = float(np.sum(weight * du * dv) / total)
    cov = np.array([[sxx, sxy], [sxy, syy]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    principal = eigvecs[:, order[0]]
    theta = float(np.arctan2(principal[1], principal[0]))
    elongation = float((eigvals[0] - eigvals[1]) / (eigvals[0] + eigvals[1])) if eigvals[0] + eigvals[1] > 1e-9 else 0.0
    return cu, cv, float(np.cos(2 * theta)), float(np.sin(2 * theta)), elongation, total


def extract_fundus_features(gray: np.ndarray, ilm_mask: np.ndarray, vessel_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uu, vv = _normalized_coords(gray.shape)

    ilm_u, ilm_v, ilm_cos2, ilm_sin2, ilm_elong, ilm_total = _orientation_features(ilm_mask, uu, vv)
    ilm_area_fraction = float(ilm_mask.mean())

    vessel_u, vessel_v, vessel_cos2, vessel_sin2, vessel_elong, _ = _orientation_features(vessel_mask, uu, vv)
    vessel_density = float(vessel_mask.mean())

    gray_f = gray.astype(np.float32)
    if ilm_mask.any():
        in_ilm = gray_f[ilm_mask]
        mean_intensity = float(in_ilm.mean())
        std_intensity = float(in_ilm.std())
        bright_thresh = np.percentile(in_ilm, BRIGHT_PERCENTILE)
        bright_weight = np.where(ilm_mask & (gray_f >= bright_thresh), gray_f, 0.0)
    else:
        mean_intensity = std_intensity = 0.0
        bright_weight = np.zeros_like(gray_f)
    bright_u, bright_v, _ = _weighted_centroid(bright_weight, uu, vv)

    compact = np.array([
        ilm_area_fraction, ilm_u, ilm_v, ilm_cos2, ilm_sin2, ilm_elong,
        vessel_density, vessel_u, vessel_v, vessel_cos2, vessel_sin2, vessel_elong,
        bright_u, bright_v, mean_intensity, std_intensity,
    ], dtype=np.float64)
    assert compact.shape[0] == len(FUNDUS_COMPACT_FEATURE_NAMES)

    ilm_grid = block_nanmean(ilm_mask.astype(np.float32), GRID, GRID)
    vessel_grid = block_nanmean(vessel_mask.astype(np.float32), GRID, GRID)
    gray_grid = block_nanmean(gray_f / 255.0, GRID, GRID)
    grid = np.concatenate([ilm_grid.ravel(), vessel_grid.ravel(), gray_grid.ravel()]).astype(np.float64)
    return compact, grid


# --------------------------------------------------------------------------
# 3. Dataset combinado: features OCT (cache) + features fundus (nuevas) + poses
# --------------------------------------------------------------------------

def load_oct_cache(cache_path: Path) -> dict:
    payload = np.load(cache_path, allow_pickle=False)
    return {
        "X_compact": payload["X_compact"], "X_grid": payload["X_grid"],
        "gt_matrices": payload["gt_matrices"],
        "scenarios": payload["scenarios"].astype(str), "case_ids": payload["case_ids"].astype(str),
    }


def build_pose_dataset(oct_cache: dict, annotations_root: Path, fundus_root: Path,
                       fundus_cache_path: Path, refresh: bool, log_every: int) -> dict:
    case_ids = oct_cache["case_ids"]
    n = len(case_ids)

    ioct_quat = np.zeros((n, 4)); eye_quat = np.zeros((n, 4))
    for i, case_id in enumerate(case_ids):
        scenario, frame_id = case_id.split("/")
        data = json.loads((annotations_root / scenario / f"{frame_id}.json").read_text(encoding="utf-8"))
        ioct_quat[i] = data["iOCT Microscope"]["Spatial"]["Rotation"]
        eye_quat[i] = data["Eyeball"]["Spatial"]["Rotation"]

    if fundus_cache_path.exists() and not refresh:
        print(f"Cargando cache de features de fundus desde {fundus_cache_path}", flush=True)
        payload = np.load(fundus_cache_path, allow_pickle=False)
        fundus_compact, fundus_grid = payload["X_compact"], payload["X_grid"]
        assert len(fundus_compact) == n, "cache de fundus desalineada con el cache de OCT"
    else:
        print(f"Extrayendo features de fundus para {n} casos...", flush=True)
        reader = FundusReader(fundus_root)
        fundus_compact = np.zeros((n, len(FUNDUS_COMPACT_FEATURE_NAMES)), dtype=np.float64)
        fundus_grid = np.zeros((n, 3 * GRID * GRID), dtype=np.float64)
        started = time.monotonic()
        for i, case_id in enumerate(case_ids):
            scenario, frame_id = case_id.split("/")
            gray, ilm_mask, vessel_mask = reader.read_case(scenario, frame_id)
            compact, grid = extract_fundus_features(gray, ilm_mask, vessel_mask)
            fundus_compact[i] = compact
            fundus_grid[i] = grid
            if (i + 1) % log_every == 0:
                elapsed = time.monotonic() - started
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else float("nan")
                print(f"[fundus {i + 1}/{n}] {rate:.1f} casos/min", flush=True)
        reader.close()
        fundus_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fundus_cache_path, X_compact=fundus_compact, X_grid=fundus_grid,
                            case_ids=case_ids)
        print(f"Features de fundus cacheadas en {fundus_cache_path}", flush=True)

    return {
        "case_ids": case_ids, "scenarios": oct_cache["scenarios"], "gt_matrices": oct_cache["gt_matrices"],
        "oct_compact": oct_cache["X_compact"], "oct_grid": oct_cache["X_grid"],
        "fundus_compact": fundus_compact, "fundus_grid": fundus_grid,
        "ioct_quat": ioct_quat, "eye_quat": eye_quat,
    }


# --------------------------------------------------------------------------
# 4. Regresion GroupKFold de cuaterniones + control de barajado
# --------------------------------------------------------------------------

def run_quaternion_regression(X_sets: dict, Y: np.ndarray, scenarios: np.ndarray, seed: int,
                              label: str) -> dict:
    splits = group_kfold_splits(scenarios, N_SPLITS)
    models = make_models(seed)
    results = {}
    best_key, best_r2 = None, -np.inf
    for fs_name, X in X_sets.items():
        for model_name, model_fn in models.items():
            key = f"{fs_name}/{model_name}"
            t0 = time.monotonic()
            oof = oof_predict(X, Y, splits, model_fn)
            elapsed = time.monotonic() - t0
            r2_joint = r2_score(Y, oof, multioutput="uniform_average")
            r2_components = [float(r2_score(Y[:, j], oof[:, j])) for j in range(Y.shape[1])]
            results[key] = {"r2_joint": float(r2_joint), "r2_components": r2_components,
                            "oof": oof, "fit_seconds": elapsed, "n_features": X.shape[1]}
            print(f"[{label}][{key}] R2_joint={r2_joint:.4f} R2_comp={np.round(r2_components,4).tolist()} "
                  f"[{elapsed:.1f}s]", flush=True)
            if r2_joint > best_r2:
                best_r2, best_key = r2_joint, key

    fs_name, model_name = best_key.split("/")
    X_best = X_sets[fs_name]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(Y))
    X_shuffled = X_best[permutation]
    oof_shuffled = oof_predict(X_shuffled, Y, splits, models[model_name])
    r2_shuffled = float(r2_score(Y, oof_shuffled, multioutput="uniform_average"))
    print(f"[{label}][control shuffle {best_key}] R2_joint={r2_shuffled:.4f} (vs {best_r2:.4f} sin barajar)",
          flush=True)

    return {"per_model": results, "best_key": best_key, "best_r2": best_r2,
            "shuffle_r2": r2_shuffled, "splits": splits}


# --------------------------------------------------------------------------
# 5. Composicion via la regresion sustituta + barrido de precision angular
# --------------------------------------------------------------------------

def matrix_from_quaternions(ioct_q: np.ndarray, eye_q: np.ndarray, poly: PolynomialFeatures,
                            surrogate: LinearRegression) -> np.ndarray:
    X = np.concatenate([ioct_q, eye_q], axis=1)
    params = surrogate.predict(poly.transform(X))
    n = len(params)
    matrices = np.zeros((n, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = params[:, 0]; matrices[:, 0, 1] = params[:, 1]; matrices[:, 0, 2] = params[:, 2]
    matrices[:, 1, 0] = params[:, 3]; matrices[:, 1, 1] = params[:, 4]; matrices[:, 1, 2] = params[:, 5]
    matrices[:, 2, 2] = 1.0
    return matrices


def random_small_rotation_quats(rng: np.random.Generator, n: int, angle_deg: float) -> np.ndarray:
    """Cuaterniones de una rotacion aleatoria de magnitud angular fija
    (radianes = angle_deg) alrededor de un eje 3D uniforme."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.radians(angle_deg) / 2.0
    return np.concatenate([axis * np.sin(half), np.full((n, 1), np.cos(half))], axis=1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """xyzw * xyzw -> xyzw (formato usado en las anotaciones, confirmado por
    norma unitaria; Opmi=[0.70711,0,0,0.70711] es 90 grados en torno a X con
    w en la ultima posicion)."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=1)


def precision_sweep(ioct_quat: np.ndarray, eye_quat: np.ndarray, gt_matrices: np.ndarray,
                    poly: PolynomialFeatures, surrogate: LinearRegression, seed: int,
                    angles_deg: list[float], trials_per_angle: int = 5,
                    perturb_ioct: bool = True, perturb_eye: bool = True,
                    label: str = "ambos") -> list[dict]:
    """Perturba uno o ambos cuaterniones con la misma magnitud angular (ruido
    independiente por caso) y mide el corner-AUC resultante al recomponer via
    la regresion sustituta. `trials_per_angle` promedia sobre varias
    direcciones de ruido aleatorias para no depender de una unica semilla.
    `perturb_ioct`/`perturb_eye` permiten aislar la sensibilidad de cada
    objeto (deja el otro en su valor GT exacto)."""
    rng = np.random.default_rng(seed)
    n = len(ioct_quat)
    rows = []
    for angle in angles_deg:
        aucs = []
        for trial in range(trials_per_angle):
            ioct_noisy, eye_noisy = ioct_quat, eye_quat
            if angle > 0.0 and perturb_ioct:
                noise_ioct = random_small_rotation_quats(rng, n, angle)
                ioct_noisy = quat_multiply(noise_ioct, ioct_quat)
            if angle > 0.0 and perturb_eye:
                noise_eye = random_small_rotation_quats(rng, n, angle)
                eye_noisy = quat_multiply(noise_eye, eye_quat)
            pred_matrices = matrix_from_quaternions(ioct_noisy, eye_noisy, poly, surrogate)
            errors = corner_error(pred_matrices, gt_matrices)
            aucs.append(corner_auc(errors))
            if angle == 0.0:
                break  # sin ruido no hay nada que promediar sobre "trials"
        rows.append({"angle_deg": angle, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))})
        print(f"[precision sweep {label}] angulo={angle:.2f}deg AUC={np.mean(aucs):.4f} +- {np.std(aucs):.4f}",
              flush=True)
    return rows


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------

def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_pose_report(results: dict, output_path: Path, args: argparse.Namespace) -> None:
    v = results["verification"]
    n = results["n_cases"]
    ioct_res = results["ioct_from_oct"]
    eye_res = results["eye_from_fundus"]
    combo = results["combo_eval"]
    ceiling_auc = combo["gt_both (sanity: reproduce la matriz real via el sustituto, sin prediccion)"]["auc"]

    lines: list[str] = []
    lines += [
        "# Task 2 como estimacion de pose (T2-98) -- iOCT.Rotation y Eyeball.Rotation",
        "",
        f"Reformulacion del coordinador, verificada de forma INDEPENDIENTE en este script "
        f"(`verify_hallazgo`) sobre los {v['n_cases']} casos de train: `Opmi.Spatial` es constante, "
        "`Eyeball.Translation` es constante (cero), y solo dos cuaterniones varian.",
        "",
        "| campo | std por componente |", "|---|---|",
        f"| `iOCT.Rotation` | {np.round(v['ioct_rotation_std'],4).tolist()} |",
        f"| `Eyeball.Rotation` | {np.round(v['eyeball_rotation_std'],4).tolist()} |",
        f"| `Eyeball.Translation` | {np.round(v['eyeball_translation_std'],4).tolist()} (constante, cero) |",
        f"| `Opmi.Rotation` | {np.round(v['opmi_rotation_std'],4).tolist()} (constante) |",
        "",
        "Regresion cuadratica de los 8 componentes de ambos cuaterniones contra los 6 parametros "
        f"de la matriz GT (ajustada sobre los {v['n_cases']} casos, como aproximacion de la geometria "
        "fija del simulador, NO la formula exacta -- ver limitacion abajo): "
        f"R2 por parametro = {np.round(v['surrogate_quadratic_r2_per_matrix_param'],3).tolist()}, "
        f"MAE por parametro = {np.round(v['surrogate_quadratic_mae_per_matrix_param'],2).tolist()} px.",
        "",
        "**Limitacion central de este informe**: la regresion sustituta NO es la formula geometrica "
        "exacta (proyeccion 3D real del simulador con los cuaterniones e intrinsecos de camara) -- "
        "es una aproximacion polinomica. Eso le pone un TECHO DURO a todo lo que sigue: incluso "
        "alimentandola con los cuaterniones GT exactos (sin ningun error de prediccion), el AUC que "
        f"produce es **{fmt(ceiling_auc)}**, no 1.0. Todos los numeros de AUC de este informe deben "
        "leerse como una COTA INFERIOR de lo que la via de estimacion de pose podria lograr con la "
        "composicion geometrica exacta -- no como el techo real de la via.",
        "",
        "## 1. iOCT.Rotation desde el VOLUMEN OCT (features reusadas de T2-96)",
        "",
        "Mismas features del volumen (perfil de espesor Ilm-Rpe, densidad de vasos e instrumento, "
        "compactas + grid 8x8) que en `experiments/96-t2-oct-self-localisation/`, solo cambia el "
        "target: de (tx,ty) a los 4 componentes del cuaternion `iOCT.Rotation`.",
        "",
        "| feature set / modelo | R2 conjunto | R2 por componente [x,y,z,w] | tiempo |",
        "|---|---:|---|---:|",
    ]
    for key, entry in sorted(ioct_res["per_model"].items(), key=lambda kv: -kv[1]["r2_joint"]):
        marker = " **<- mejor**" if key == ioct_res["best_key"] else ""
        lines.append(f"| {key}{marker} | {fmt(entry['r2_joint'])} | "
                     f"{np.round(entry['r2_components'],4).tolist()} | {entry['fit_seconds']:.1f}s |")
    lines += [
        "",
        f"Control de barajado (mejor modelo, `{ioct_res['best_key']}`): R2 con features de OCT "
        f"barajadas entre casos = **{fmt(ioct_res['shuffle_r2'])}** (vs. {fmt(ioct_res['best_r2'])} sin "
        "barajar).",
        "",
        "## 2. Eyeball.Rotation desde el FUNDUS (features nuevas, monomodal)",
        "",
        "Features nuevas del lado microscopio (nunca del OCT): mascara de retina visible (`ilm.png`, "
        "area/centroide/orientacion), mascara de vasos (`arteriesorveins.png`, densidad/centroide/"
        "orientacion), y brillo (proxy del disco optico -- percentil 95 de intensidad dentro de la "
        "retina visible). Compactas + grid 8x8.",
        "",
        "| feature set / modelo | R2 conjunto | R2 por componente [x,y,z,w] | tiempo |",
        "|---|---:|---|---:|",
    ]
    for key, entry in sorted(eye_res["per_model"].items(), key=lambda kv: -kv[1]["r2_joint"]):
        marker = " **<- mejor**" if key == eye_res["best_key"] else ""
        lines.append(f"| {key}{marker} | {fmt(entry['r2_joint'])} | "
                     f"{np.round(entry['r2_components'],4).tolist()} | {entry['fit_seconds']:.1f}s |")
    lines += [
        "",
        f"Control de barajado (mejor modelo, `{eye_res['best_key']}`): R2 con features de fundus "
        f"barajadas entre casos = **{fmt(eye_res['shuffle_r2'])}** (vs. {fmt(eye_res['best_r2'])} sin "
        "barajar).",
        "",
        "## Contribucion al AUC oficial (via la regresion sustituta, sujeta a la limitacion de arriba)",
        "",
        "| combinacion | AUC | error medio (px) | error mediano (px) |", "|---|---:|---:|---:|",
    ]
    for name, ev in combo.items():
        lines.append(f"| {name} | {fmt(ev['auc'])} | {fmt(ev['mean_error'],2)} | {fmt(ev['median_error'],2)} |")

    lines += [
        "",
        "## 3. Cuanta precision angular hace falta (barrido, sujeto a la misma limitacion)",
        "",
        f"Techo con cuaterniones GT exactos (angulo=0): AUC=**{fmt(ceiling_auc)}**. A partir de ahi se "
        "perturba cada cuaternion con ruido angular de magnitud creciente (eje aleatorio, 5 tiradas "
        "por magnitud) y se mide cuanto tarda en desplomarse el AUC.",
        "",
        "| angulo (deg) | AUC ambos perturbados | AUC solo iOCT | AUC solo ojo |",
        "|---:|---:|---:|---:|",
    ]
    both = {r["angle_deg"]: r["auc_mean"] for r in results["precision_sweep_both"]}
    only_ioct = {r["angle_deg"]: r["auc_mean"] for r in results["precision_sweep_ioct_only"]}
    only_eye = {r["angle_deg"]: r["auc_mean"] for r in results["precision_sweep_eye_only"]}
    for angle in sorted(both):
        lines.append(f"| {angle:g} | {fmt(both[angle])} | {fmt(only_ioct[angle])} | {fmt(only_eye[angle])} |")

    lines += [
        "",
        "`iOCT.Rotation` es mucho mas sensible que `Eyeball.Rotation`: perturbar solo el cuaternion del "
        "escaner desploma el AUC en 2-3 grados, mientras que perturbar solo el del ojo tolera bastante "
        "mas antes de colapsar. La rotacion del escaner es el eslabon critico: define que ventana de la "
        "retina capturo el volumen, y un pequeno error ahi mueve la traslacion completa fuera de la "
        "ventana de 10px que puntua la metrica.",
        "",
        "## Interpretacion",
        "",
    ]
    # OJO: seleccionar por CLAVE EXACTA, no por "pred" in name -- el nombre de la
    # entrada sanity ("sin prediccion") contiene la subcadena "pred" y contaminaba
    # el maximo con el techo del sustituto en vez de una prediccion real.
    both_pred_key = next(name for name in combo if name.startswith("pred_ioct + pred_eye"))
    ioct_only_key = next(name for name in combo if name.startswith("pred_ioct + gt_eye"))
    eye_only_key = next(name for name in combo if name.startswith("gt_ioct + pred_eye"))
    both_pred_auc = combo[both_pred_key]["auc"]
    ioct_only_auc = combo[ioct_only_key]["auc"]
    eye_only_auc = combo[eye_only_key]["auc"]
    best_combo_auc = max(both_pred_auc, ioct_only_auc, eye_only_auc)

    lines.append(
        f"AUC con AMBAS rotaciones predichas (OOF) = **{fmt(both_pred_auc)}**. Contribucion aislada: "
        f"solo iOCT predicho (ojo en GT) = **{fmt(ioct_only_auc)}**; solo ojo predicho (iOCT en GT) = "
        f"**{fmt(eye_only_auc)}**. Techo del sustituto con ambas rotaciones GT (sin ninguna prediccion) "
        f"= {fmt(ceiling_auc)}.")
    lines.append("")
    if eye_only_auc >= ioct_only_auc * 2 and eye_only_auc > 0.05:
        lines.append(
            f"**Asimetria real y contraintuitiva**: `iOCT.Rotation` tiene MEJOR R2 (seccion 1, "
            f"{fmt(ioct_res['best_r2'])}) que `Eyeball.Rotation` (seccion 2, {fmt(eye_res['best_r2'])}), "
            "pero su contribucion aislada al AUC es MENOR. Se explica por la seccion 3: el AUC es mucho "
            "mas sensible a errores en `iOCT.Rotation` (colapsa en 2-3 grados) que a errores en "
            "`Eyeball.Rotation` (tolera bastante mas), asi que el mismo nivel de error de prediccion "
            "pesa mucho mas cuando cae sobre la rotacion del escaner. La via del FUNDUS (monomodal, "
            "mas facil de mejorar) es la que mas cerca esta de aportar algo al AUC, no la del volumen "
            "OCT pese a su R2 mas alto.")
        lines.append("")
    if best_combo_auc >= 0.10:
        lines.append(f"Mejor contribucion aislada AUC={fmt(best_combo_auc)} >= 0.10 (recordar que es "
                     "COTA INFERIOR por la limitacion del sustituto): la via de pose es viable y vale "
                     "la pena invertir en derivar la composicion geometrica exacta.")
    else:
        lines.append(
            f"Con las features probadas aqui, incluso la mejor contribucion aislada de AUC="
            f"{fmt(best_combo_auc)} (y la combinacion de ambas predicciones, {fmt(both_pred_auc)}) "
            f"quedan muy por debajo del techo geometrico del sustituto ({fmt(ceiling_auc)}), que a su "
            "vez ya esta muy por debajo de 0.475 (lider). El cuello de botella no es la formula de "
            "composicion (aun no derivada exactamente) sino la PRECISION de la prediccion de rotacion "
            "desde las features de apariencia -- el barrido de la seccion 3 muestra que hace falta "
            "precision sub-grado en `iOCT.Rotation` y de unos pocos grados en `Eyeball.Rotation`, un "
            "listón que estas features no alcanzan ni de lejos con este n y estos modelos.")

    lines += [
        "",
        "## Pendientes declarados (de `HALLAZGO.md`, no resueltos aqui)",
        "",
        "- Derivar la composicion geometrica EXACTA (proyeccion 3D del simulador con los cuaterniones e "
        "intrinsecos de camara reales) en vez de la regresion cuadratica sustituta -- resolveria la "
        "limitacion central de este informe y probablemente subiria el techo bastante por encima de "
        f"{fmt(ceiling_auc)}.",
        "- Confirmar si Task 1 anota `eye pose` con el mismo convenio (50x mas frames disponibles para "
        "supervisar `Eyeball.Rotation` desde el fundus, si es cierto).",
        "",
        "## Notas de implementacion",
        "",
        "- Reusa el cache de features OCT de `measure_task2_oct_self_localisation.py` "
        f"(`{args.oct_cache}`) -- no se releyeron los volumenes.",
        f"- Features de fundus nuevas, cacheadas en `{args.fundus_cache}`.",
        f"- GroupKFold leave-one-scenario-out, n_splits={N_SPLITS}, igual que en T2-96.",
        "- `evaluate_task2_cases`/`corner_auc`/`corner_error` de `src/fido/eval_task2.py` y "
        "`src/fido/geometry.py` -- misma metrica oficial que el resto del proyecto.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe de pose escrito en {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, default=REPO_ROOT / "data/_annotations/Task 2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/Task 2")
    parser.add_argument("--oct-cache", type=Path,
                        default=REPO_ROOT / "experiments/96-t2-oct-self-localisation/features_cache.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/98-t2-pose-estimation")
    parser.add_argument("--fundus-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh-fundus", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.fundus_cache is None:
        args.fundus_cache = args.output_dir / "fundus_features_cache.npz"

    require_task2_training_root(args.data_root)

    print("=== Verificando hallazgo T2-98 de forma independiente ===", flush=True)
    verification = verify_hallazgo(args.annotations_root)
    print(f"n={verification['n_cases']} ioct_rotation_std={np.round(verification['ioct_rotation_std'],4)} "
          f"eyeball_rotation_std={np.round(verification['eyeball_rotation_std'],4)}", flush=True)
    print(f"surrogate cuadratico R2 por parametro de matriz: "
          f"{np.round(verification['surrogate_quadratic_r2_per_matrix_param'],3)}", flush=True)

    print("=== Cargando cache de features OCT (de measure_task2_oct_self_localisation.py) ===", flush=True)
    oct_cache = load_oct_cache(args.oct_cache)
    print(f"{len(oct_cache['case_ids'])} casos en cache OCT", flush=True)

    dataset = build_pose_dataset(oct_cache, args.annotations_root, args.data_root, args.fundus_cache,
                                 args.refresh_fundus, args.log_every)
    n = len(dataset["case_ids"])
    scenarios = dataset["scenarios"]

    oct_sets = {"compact": dataset["oct_compact"],
               "full": np.concatenate([dataset["oct_compact"], dataset["oct_grid"]], axis=1)}
    fundus_sets = {"compact": dataset["fundus_compact"],
                  "full": np.concatenate([dataset["fundus_compact"], dataset["fundus_grid"]], axis=1)}

    print("=== Pregunta 1: iOCT.Rotation desde el VOLUMEN OCT ===", flush=True)
    ioct_results = run_quaternion_regression(oct_sets, dataset["ioct_quat"], scenarios, args.seed,
                                             "iOCT<-OCT")

    print("=== Pregunta 2: Eyeball.Rotation desde el FUNDUS ===", flush=True)
    eye_results = run_quaternion_regression(fundus_sets, dataset["eye_quat"], scenarios, args.seed,
                                            "Eye<-Fundus")

    poly, surrogate = verification["poly"], verification["surrogate_model"]
    gt_matrices = dataset["gt_matrices"]
    ioct_quat, eye_quat = dataset["ioct_quat"], dataset["eye_quat"]

    ioct_oof = ioct_results["per_model"][ioct_results["best_key"]]["oof"]
    eye_oof = eye_results["per_model"][eye_results["best_key"]]["oof"]

    def normalize_quats(q):
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    print("=== Componiendo matrices via la regresion sustituta ===", flush=True)
    combos = {
        "gt_both (sanity: reproduce la matriz real via el sustituto, sin prediccion)":
            matrix_from_quaternions(ioct_quat, eye_quat, poly, surrogate),
        "pred_ioct + gt_eye (contribucion del volumen OCT sola)":
            matrix_from_quaternions(normalize_quats(ioct_oof), eye_quat, poly, surrogate),
        "gt_ioct + pred_eye (contribucion del fundus solo)":
            matrix_from_quaternions(ioct_quat, normalize_quats(eye_oof), poly, surrogate),
        "pred_ioct + pred_eye (todo predicho, ambas modalidades)":
            matrix_from_quaternions(normalize_quats(ioct_oof), normalize_quats(eye_oof), poly, surrogate),
    }
    combo_eval = {}
    for name, matrices in combos.items():
        ev = evaluate_task2_cases(matrices, gt_matrices, scenarios, case_ids=dataset["case_ids"])
        combo_eval[name] = {"auc": ev["auc"], "mean_error": ev["mean_error"], "median_error": ev["median_error"]}
        print(f"[{name}] AUC={ev['auc']:.4f} mean_error={ev['mean_error']:.2f}px", flush=True)

    print("=== Pregunta 3: barrido de precision angular ===", flush=True)
    angles = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0]
    sweep_both = precision_sweep(ioct_quat, eye_quat, gt_matrices, poly, surrogate, args.seed, angles,
                                 perturb_ioct=True, perturb_eye=True, label="ambos")
    sweep_ioct_only = precision_sweep(ioct_quat, eye_quat, gt_matrices, poly, surrogate, args.seed, angles,
                                      perturb_ioct=True, perturb_eye=False, label="solo iOCT")
    sweep_eye_only = precision_sweep(ioct_quat, eye_quat, gt_matrices, poly, surrogate, args.seed, angles,
                                     perturb_ioct=False, perturb_eye=True, label="solo ojo")

    results = {
        "verification": {k: v for k, v in verification.items() if k not in ("poly", "surrogate_model")},
        "ioct_from_oct": {k: v for k, v in ioct_results.items() if k != "splits"},
        "eye_from_fundus": {k: v for k, v in eye_results.items() if k != "splits"},
        "combo_eval": combo_eval,
        "precision_sweep_both": sweep_both, "precision_sweep_ioct_only": sweep_ioct_only,
        "precision_sweep_eye_only": sweep_eye_only, "n_cases": n,
    }
    for block in (results["ioct_from_oct"]["per_model"], results["eye_from_fundus"]["per_model"]):
        for entry in block.values():
            entry.pop("oof", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"Resultados crudos en {args.output_dir / 'results.json'}", flush=True)

    write_pose_report(results, args.output_dir / "INFORME.md", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
