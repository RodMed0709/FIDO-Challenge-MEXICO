"""T2-100 -- estimar `s0`/`c_tx`/`c_ty` (los coeficientes de escala y
traslacion del modelo gnomonico, ver `experiments/98-t2-pose-estimation/
GEOMETRIA_EXACTA.md` y `experiments/99-t2-scenario-heterogeneity/INFORME.md`)
a partir de FEATURES DE IMAGEN del fundus (`Stereo Left/<frame>/
{microscope.png, visibility.png}`), en vez de recalibrarlos con el GT del
propio escenario (oraculo, no desplegable).

Contexto verificado (T2-98/99, reusado aqui sin reimplementar -- se importa
`derive_task2_geometry.py` por ruta, igual que hace
`analyze_task2_scenario_heterogeneity.py`):
  - Coeficientes GLOBALES (un solo ajuste para los 10 escenarios): AUC LOSO
    oficial pooled = 0.1512.
  - Recalibrando POR ESCENARIO los 8 numeros libres de las tres funciones
    gnomonicas (`coef_scale=[s0,s1]`, `coef_tx=[c_gx,c_gy,c0]`,
    `coef_ty=[c_gx,c_gy,c0]`) con el GT del propio escenario de test
    (oraculo, diagnostico): AUC LOSO pooled = 0.7204, mean_err 2.59px, todos
    los escenarios AUC>=0.61 (ver INFORME.md T2-99 seccion 3.3, ablacion
    "scale+tx+ty"). Nota de nomenclatura: la tarea que origino este script se
    referia a estos 8 numeros de forma abreviada como "s0, c_tx, c_ty" (los
    SIMBOLOS del intercepto de cada una de las tres funciones); se comprobo
    aqui mismo (ver INFORME.md seccion 0) que recalibrar SOLO esos 3
    interceptos (dejando las 5 pendientes en gx,gy fijas globalmente) da
    apenas AUC=0.2550 -- muy por debajo de 0.7204 -- asi que el objetivo real
    de este script es reproducir/aproximar el recalibrado de los 8 numeros,
    no solo 3 escalares.

Este script:
  1. Extrae features geometricas del fundus (mascara iluminada/no-negra via
     `visibility.png` y umbral RGB de `microscope.png`, perfil radial de
     brillo) para los 1214 casos de train, con cache en npz.
  2. Mide heterogeneidad intra- vs entre-escenario de esas features y de los
     coeficientes oraculo (por-escenario).
  3. Comprueba con un split-half bootstrap si los coeficientes oraculo son
     estables DENTRO de un escenario (soportando granularidad por-escenario)
     o si varian tanto como entre escenarios (lo que exigiria granularidad
     por-caso).
  4. Ajusta un predictor LINEAL de los 8 coeficientes a partir de una base
     fija de features de imagen (elegida a priori por razonamiento fisico,
     NO por mineria de correlaciones), bajo leave-one-scenario-out anidado
     (los 9 escenarios de entrenamiento del fold externo se usan para
     ajustar el predictor; el escenario de test nunca se ve), compone la
     matriz predicha y mide el AUC oficial end-to-end.
  5. Compara contra 0.1512 (coeficientes globales) y 0.7204 (oraculo).

Todo numpy/scipy/PIL/sklearn, local, CPU. Ejecutar:
    KMP_DUPLICATE_LIB_OK=TRUE python analysis/estimate_task2_coefficients_from_image.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats as sstats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.geometry import corner_auc, corner_error  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "derive_task2_geometry", REPO_ROOT / "analysis" / "derive_task2_geometry.py"
)
dtg = importlib.util.module_from_spec(_spec)
sys.modules["derive_task2_geometry"] = dtg
_spec.loader.exec_module(dtg)

ANNOTATIONS_ROOT = REPO_ROOT / "data" / "_annotations" / "Task 2"
IMAGE_ROOT = REPO_ROOT / "data" / "Task 2"
OUTPUT_DIR = REPO_ROOT / "experiments" / "100-t2-coefficients-from-image"
CACHE_PATH = OUTPUT_DIR / "image_features_cache.npz"

IMG_SIZE = 1024
CENTER = (IMG_SIZE / 2.0, IMG_SIZE / 2.0)  # (row, col) del centro geometrico del canvas

FEATURE_NAMES = [
    # -- de la mascara de visibilidad (visibility.png, binaria, oficial) --
    "vis_area_frac", "vis_centroid_y", "vis_centroid_x", "vis_r_equiv",
    "vis_bbox_ymin", "vis_bbox_ymax", "vis_bbox_xmin", "vis_bbox_xmax",
    "vis_bbox_h", "vis_bbox_w", "vis_bbox_cy", "vis_bbox_cx",
    "vis_eccentricity", "vis_edge_r_p95",
    # -- de la mascara no-negra del propio microscope.png (redundante, cross-check) --
    "nb_area_frac", "nb_centroid_y", "nb_centroid_x", "nb_r_equiv",
    "nb_bbox_ymin", "nb_bbox_ymax", "nb_bbox_xmin", "nb_bbox_xmax",
    "nb_bbox_h", "nb_bbox_w", "nb_bbox_cy", "nb_bbox_cx",
    "nb_eccentricity", "nb_edge_r_p95",
    # -- perfil radial de brillo (grises, desde el centro del canvas fijo) --
    "profile_edge_r50",
]


# --------------------------------------------------------------------------
# 1. Extraccion de features de imagen
# --------------------------------------------------------------------------

def mask_geometry_features(mask: np.ndarray, prefix: str) -> dict:
    h, w = mask.shape
    area_frac = float(mask.mean())
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        nan = float("nan")
        return {f"{prefix}_{k}": nan for k in
                ["area_frac", "centroid_y", "centroid_x", "r_equiv",
                 "bbox_ymin", "bbox_ymax", "bbox_xmin", "bbox_xmax",
                 "bbox_h", "bbox_w", "bbox_cy", "bbox_cx",
                 "eccentricity", "edge_r_p95"]}
    cy, cx = float(ys.mean()), float(xs.mean())
    r_equiv = float(np.sqrt(area_frac * h * w / np.pi))
    ymin, ymax, xmin, xmax = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    # excentricidad: autovalores de la matriz de covarianza de los pixeles de la mascara
    cov = np.cov(np.stack([ys - cy, xs - cx]))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    ecc = float(np.sqrt(1.0 - eigvals[0] / eigvals[1]))
    dist = np.hypot(ys - cy, xs - cx)
    edge_r_p95 = float(np.percentile(dist, 95))
    return {
        f"{prefix}_area_frac": area_frac, f"{prefix}_centroid_y": cy, f"{prefix}_centroid_x": cx,
        f"{prefix}_r_equiv": r_equiv,
        f"{prefix}_bbox_ymin": float(ymin), f"{prefix}_bbox_ymax": float(ymax),
        f"{prefix}_bbox_xmin": float(xmin), f"{prefix}_bbox_xmax": float(xmax),
        f"{prefix}_bbox_h": float(ymax - ymin), f"{prefix}_bbox_w": float(xmax - xmin),
        f"{prefix}_bbox_cy": float((ymin + ymax) / 2.0), f"{prefix}_bbox_cx": float((xmin + xmax) / 2.0),
        f"{prefix}_eccentricity": ecc, f"{prefix}_edge_r_p95": edge_r_p95,
    }


def radial_profile_edge(gray: np.ndarray, center: tuple[float, float], n_bins: int = 120) -> float:
    """Radio (px) donde la intensidad media cae por debajo de la mitad del
    nivel "interior" (mediana del 20% de bins mas cercanos al centro) --
    proxy del borde del vineteado optico, independiente de cualquier umbral
    binario duro."""
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.hypot(ys - center[0], xs - center[1])
    max_r = float(dist.max())
    bins = np.linspace(0, max_r, n_bins + 1)
    idx = np.clip(np.digitize(dist.ravel(), bins) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=gray.ravel().astype(np.float64), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    prof = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    interior = np.nanmedian(prof[: max(1, n_bins // 5)])
    if interior <= 0:
        return float("nan")
    below = np.where(prof < 0.5 * interior)[0]
    centers = (bins[:-1] + bins[1:]) / 2.0
    if len(below) == 0:
        return float(max_r)
    return float(centers[below[0]])


def extract_case_features(microscope_rgb: np.ndarray, visibility_gray: np.ndarray) -> dict:
    feats = {}
    vis_mask = visibility_gray > 128
    feats.update(mask_geometry_features(vis_mask, "vis"))
    nb_mask = microscope_rgb.astype(np.int32).sum(axis=2) > 10
    feats.update(mask_geometry_features(nb_mask, "nb"))
    gray = microscope_rgb.astype(np.float64).mean(axis=2)
    feats["profile_edge_r50"] = radial_profile_edge(gray, CENTER)
    return feats


def open_case_images(scenario: str, frame_id: str, zip_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    extracted_dir = IMAGE_ROOT / f"Scenario_{scenario}" / "Stereo Left" / frame_id
    if extracted_dir.is_dir():
        microscope = np.array(Image.open(extracted_dir / "microscope.png").convert("RGB"))
        visibility = np.array(Image.open(extracted_dir / "visibility.png").convert("L"))
        return microscope, visibility
    key = scenario
    if key not in zip_cache:
        zip_cache[key] = zipfile.ZipFile(IMAGE_ROOT / f"Scenario_{scenario}.zip")
    z = zip_cache[key]
    micro_bytes = z.read(f"Stereo Left/{frame_id}/microscope.png")
    vis_bytes = z.read(f"Stereo Left/{frame_id}/visibility.png")
    microscope = np.array(Image.open(io.BytesIO(micro_bytes)).convert("RGB"))
    visibility = np.array(Image.open(io.BytesIO(vis_bytes)).convert("L"))
    return microscope, visibility


def build_or_load_feature_cache(case_id: np.ndarray, scenario: np.ndarray, force: bool = False) -> np.ndarray:
    if CACHE_PATH.exists() and not force:
        payload = np.load(CACHE_PATH, allow_pickle=True)
        if list(payload["case_id"]) == list(case_id):
            print(f"[cache] usando features de imagen cacheadas en {CACHE_PATH}")
            return payload["X"]
        print("[cache] cache no coincide con el dataset actual -- se reconstruye")

    n = len(case_id)
    X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    zip_cache: dict = {}
    print(f"[extract] extrayendo features de imagen para {n} casos ...")
    for i, (cid, scen) in enumerate(zip(case_id, scenario)):
        frame_id = cid.split("/")[-1]
        microscope, visibility = open_case_images(scen, frame_id, zip_cache)
        feats = extract_case_features(microscope, visibility)
        X[i] = [feats[name] for name in FEATURE_NAMES]
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n}")
    for zf in zip_cache.values():
        zf.close()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, X=X, case_id=case_id, feature_names=np.array(FEATURE_NAMES))
    print(f"[extract] cache guardada en {CACHE_PATH}")
    return X


# --------------------------------------------------------------------------
# 2. Coeficientes oraculo por escenario (8 numeros: s0,s1,c_gx_tx,c_gy_tx,
#    c0_tx,c_gx_ty,c_gy_ty,c0_ty) -- reusa dtg.fit_model, un ajuste por
#    escenario con SOLO los datos de ese escenario.
# --------------------------------------------------------------------------

COEF_NAMES = ["s0", "s1", "c_gx_tx", "c_gy_tx", "c0_tx", "c_gx_ty", "c_gy_ty", "c0_ty"]


def params_to_vector(params: dict) -> np.ndarray:
    return np.array([
        params["coef_scale"][0], params["coef_scale"][1],
        params["coef_tx"][0], params["coef_tx"][1], params["coef_tx"][2],
        params["coef_ty"][0], params["coef_ty"][1], params["coef_ty"][2],
    ])


def vector_to_params(vec: np.ndarray, theta0: float) -> dict:
    return dict(theta0=theta0,
                coef_scale=np.array(vec[0:2]),
                coef_tx=np.array(vec[2:5]),
                coef_ty=np.array(vec[5:8]))


def oracle_coefficients_per_scenario(ioct_q: np.ndarray, gt: np.ndarray, scenario: np.ndarray,
                                      v0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> dict:
    out = {}
    for scen in sorted(set(scenario)):
        idx = np.where(scenario == scen)[0]
        params = dtg.fit_model(ioct_q[idx], gt[idx], v0, e1, e2)
        out[scen] = params_to_vector(params)
    return out


# --------------------------------------------------------------------------
# 3. Heterogeneidad intra- vs entre-escenario (features de imagen y
#    coeficientes oraculo)
# --------------------------------------------------------------------------

def within_between_dispersion(values: np.ndarray, scenario: np.ndarray) -> dict:
    scens = sorted(set(scenario))
    means, within_sds = [], []
    for scen in scens:
        v = values[scenario == scen]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        means.append(v.mean())
        within_sds.append(v.std())
    means = np.asarray(means)
    return dict(between_sd=float(means.std()), max_within_sd=float(max(within_sds)),
                ratio=float(means.std() / max(within_sds, default=1.0) if max(within_sds, default=0) > 0 else float("nan")))


# --------------------------------------------------------------------------
# 4. Split-half bootstrap: estabilidad de los coeficientes oraculo DENTRO
#    de un escenario (task item 3 -- por-escenario vs por-caso)
# --------------------------------------------------------------------------

def split_half_stability(ioct_q, gt, scenario, v0, e1, e2, n_trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scens = sorted(set(scenario))
    oracle_full = oracle_coefficients_per_scenario(ioct_q, gt, scenario, v0, e1, e2)
    between_sd = np.array([oracle_full[s] for s in scens]).std(axis=0)  # (8,)

    split_diffs = []  # |mitad_a - mitad_b| por coeficiente, por escenario, por trial
    for scen in scens:
        idx = np.where(scenario == scen)[0]
        if len(idx) < 20:
            continue
        for _ in range(n_trials):
            perm = rng.permutation(idx)
            half = len(perm) // 2
            idx_a, idx_b = perm[:half], perm[half:]
            try:
                pa = params_to_vector(dtg.fit_model(ioct_q[idx_a], gt[idx_a], v0, e1, e2))
                pb = params_to_vector(dtg.fit_model(ioct_q[idx_b], gt[idx_b], v0, e1, e2))
            except np.linalg.LinAlgError:
                continue
            split_diffs.append(np.abs(pa - pb))
    split_diffs = np.array(split_diffs)  # (n, 8)
    within_case_spread = split_diffs.mean(axis=0)  # media de |diferencia entre dos mitades aleatorias|
    return dict(
        coef_names=COEF_NAMES,
        between_scenario_sd=between_sd.tolist(),
        mean_split_half_abs_diff=within_case_spread.tolist(),
        ratio_between_over_splithalf=(between_sd / np.maximum(within_case_spread, 1e-9)).tolist(),
    )


# --------------------------------------------------------------------------
# 5. Predictor lineal de los 8 coeficientes desde features de imagen,
#    LOSO anidado, evaluado con la metrica oficial end-to-end.
# --------------------------------------------------------------------------

# Base de features elegida A PRIORI por razonamiento fisico (no minada por
# correlacion): intercepto + radio equivalente de la mascara de visibilidad
# (proxy del "zoom") + centroide y/x de la mascara de visibilidad (proxy del
# offset del centro de proyeccion). 4 numeros libres por objetivo, ajustados
# sobre 9 puntos (escenarios) por fold LOSO externo.
PRIMARY_FEATURES = ["vis_r_equiv", "vis_centroid_y", "vis_centroid_x"]


def build_design_matrix(X_scenario_mean: dict, scens: list[str], feature_names: list[str],
                          all_feature_names: list[str]) -> np.ndarray:
    cols = [all_feature_names.index(f) for f in feature_names]
    rows = [X_scenario_mean[s][cols] for s in scens]
    F = np.stack(rows, axis=0)
    return np.concatenate([np.ones((len(scens), 1)), F], axis=1)  # (n_scen, 1+len(feature_names))


def fit_predict_linear(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Minimos cuadrados con pequena regularizacion ridge (evita
    inestabilidad numerica con pocos puntos), un ajuste independiente por
    columna de salida."""
    lam = 1e-6
    XtX = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    coef = np.linalg.solve(XtX, Xtr.T @ Ytr)
    return Xte @ coef


def run_image_based_loso(data: dict, X_img: np.ndarray, feature_names_used: list[str]) -> dict:
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    n = len(ioct_q)
    all_errors = np.zeros(n)
    per_scenario = {}
    predicted_coefs = {}
    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        train_scens = sorted(set(scenario[train_idx]))

        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[train_idx])
        params_train = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)
        oracle_train = oracle_coefficients_per_scenario(ioct_q[train_idx], gt[train_idx],
                                                          scenario[train_idx], v0, e1, e2)

        X_scen_mean = {}
        for s in train_scens + [scen]:
            idx_s = np.where(scenario == s)[0]
            X_scen_mean[s] = np.nanmean(X_img[idx_s], axis=0)

        Xtr = build_design_matrix(X_scen_mean, train_scens, feature_names_used, FEATURE_NAMES)
        Ytr = np.stack([oracle_train[s] for s in train_scens], axis=0)
        Xte = build_design_matrix(X_scen_mean, [scen], feature_names_used, FEATURE_NAMES)
        y_pred = fit_predict_linear(Xtr, Ytr, Xte)[0]
        predicted_coefs[scen] = y_pred.tolist()

        params_pred = vector_to_params(y_pred, params_train["theta0"])
        pred = dtg.predict_matrix(ioct_q[test_idx], v0, e1, e2, params_pred)
        errs = corner_error(pred, gt[test_idx])
        all_errors[test_idx] = errs
        per_scenario[scen] = dict(n=int(len(test_idx)), mean_error=float(errs.mean()),
                                   median_error=float(np.median(errs)), auc=float(corner_auc(errs)))
    return dict(per_scenario=per_scenario, pooled_auc=float(corner_auc(all_errors)),
                pooled_mean_error=float(all_errors.mean()),
                pooled_median_error=float(np.median(all_errors)),
                predicted_coefs=predicted_coefs)


def run_intercept_only_oracle(data: dict) -> dict:
    """Referencia citada en el docstring del modulo: recalibrar SOLO los 3
    interceptos (s0, c0_tx, c0_ty) con el GT del propio escenario, dejando
    las 5 pendientes en gx,gy fijas globalmente (del fold LOSO). Cuantifica
    cuanto del AUC=0.7204 depende de las pendientes, no solo de los
    interceptos."""
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    n = len(ioct_q)
    all_err = np.zeros(n)
    per_scen = {}
    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[train_idx])
        params_train = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)
        gx, gy, roll, _ = dtg.geometric_features(ioct_q[test_idx], v0, e1, e2)
        theta_gt, scale_gt, tx_gt, ty_gt = dtg.gt_similarity_params(gt[test_idx])
        r2 = gx * gx + gy * gy
        s0_new = np.mean(scale_gt - params_train["coef_scale"][1] * r2)
        c0_tx_new = np.mean(tx_gt - params_train["coef_tx"][0] * gx - params_train["coef_tx"][1] * gy)
        c0_ty_new = np.mean(ty_gt - params_train["coef_ty"][0] * gx - params_train["coef_ty"][1] * gy)
        params = dict(params_train)
        params["coef_scale"] = np.array([s0_new, params_train["coef_scale"][1]])
        params["coef_tx"] = np.array([params_train["coef_tx"][0], params_train["coef_tx"][1], c0_tx_new])
        params["coef_ty"] = np.array([params_train["coef_ty"][0], params_train["coef_ty"][1], c0_ty_new])
        pred = dtg.predict_matrix(ioct_q[test_idx], v0, e1, e2, params)
        errs = corner_error(pred, gt[test_idx])
        all_err[test_idx] = errs
        per_scen[scen] = float(corner_auc(errs))
    return dict(per_scenario=per_scen, pooled_auc=float(corner_auc(all_err)), pooled_mean_error=float(all_err.mean()))


def run_null_permutation_check(data: dict, X_img: np.ndarray, n_trials: int, seed: int) -> dict:
    """Chequeo de escepticismo obligatorio con n=10 escenarios: sustituye la
    feature de imagen por RUIDO GAUSSIANO puro (misma forma, mismo pipeline
    de agregacion por escenario + regresion lineal 2-parametros + LOSO
    anidado) y mide que AUC end-to-end se obtiene SOLO por azar, para poder
    comparar los resultados con features reales contra esta distribucion
    nula en vez de contra 0 a secas."""
    rng = np.random.default_rng(seed)
    n = X_img.shape[0]
    aucs = []
    noise_name = "__noise__"
    for _ in range(n_trials):
        noise_col = rng.normal(size=(n, 1))
        X_aug = np.concatenate([X_img, noise_col], axis=1)
        FEATURE_NAMES.append(noise_name)
        try:
            r = run_image_based_loso(data, X_aug, [noise_name])
        finally:
            FEATURE_NAMES.pop()
        aucs.append(r["pooled_auc"])
    aucs = np.asarray(aucs)
    return dict(n_trials=n_trials, auc_mean=float(aucs.mean()), auc_std=float(aucs.std()),
                auc_min=float(aucs.min()), auc_max=float(aucs.max()), all_aucs=aucs.tolist())


def run_full_oracle_scenario(data: dict) -> dict:
    """Referencia: recalibrar los 8 numeros con el GT del propio escenario
    (oraculo, no desplegable) -- reproduce el 0.7204 de T2-99 seccion 3.3
    ('scale+tx+ty') como chequeo de consistencia."""
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    n = len(ioct_q)
    all_err = np.zeros(n)
    per_scen = {}
    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[train_idx])
        params_train = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)
        params_oracle = dtg.fit_model(ioct_q[test_idx], gt[test_idx], v0, e1, e2)
        params = dict(params_train)
        params["coef_scale"] = params_oracle["coef_scale"]
        params["coef_tx"] = params_oracle["coef_tx"]
        params["coef_ty"] = params_oracle["coef_ty"]
        pred = dtg.predict_matrix(ioct_q[test_idx], v0, e1, e2, params)
        errs = corner_error(pred, gt[test_idx])
        all_err[test_idx] = errs
        per_scen[scen] = float(corner_auc(errs))
    return dict(per_scenario=per_scen, pooled_auc=float(corner_auc(all_err)), pooled_mean_error=float(all_err.mean()))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = dtg.load_dataset(ANNOTATIONS_ROOT)
    n = len(data["ioct_q"])
    scenario = data["scenario"]
    print(f"n_cases={n}  scenarios={sorted(set(scenario))}")

    X_img = build_or_load_feature_cache(data["case_id"], scenario, force=args.force_cache)

    # --- referencias verificadas (recomputadas aqui como chequeo de consistencia) ---
    print("\n=== Referencias (recomputadas, deben coincidir con T2-98/99) ===")
    loso_global = dtg.run_loso(data)
    print(f"Coeficientes GLOBALES (sin recalibrar): AUC={loso_global['pooled_auc']:.4f} "
          f"mean_err={loso_global['pooled_mean_error']:.2f}px  [ref: 0.1512]")
    intercept_oracle = run_intercept_only_oracle(data)
    print(f"Oraculo SOLO 3 interceptos (s0,c0_tx,c0_ty), pendientes globales: "
          f"AUC={intercept_oracle['pooled_auc']:.4f} mean_err={intercept_oracle['pooled_mean_error']:.2f}px")
    full_oracle = run_full_oracle_scenario(data)
    print(f"Oraculo COMPLETO 8 numeros por escenario (scale+tx+ty): "
          f"AUC={full_oracle['pooled_auc']:.4f} mean_err={full_oracle['pooled_mean_error']:.2f}px  [ref: 0.7204]")

    # --- heterogeneidad intra vs entre escenario: features de imagen ---
    print("\n=== Heterogeneidad intra- vs entre-escenario: features de imagen ===")
    feat_hetero = {}
    for j, name in enumerate(FEATURE_NAMES):
        feat_hetero[name] = within_between_dispersion(X_img[:, j], scenario)
    for name in ["vis_area_frac", "vis_r_equiv", "vis_centroid_y", "vis_centroid_x",
                 "nb_area_frac", "nb_r_equiv", "profile_edge_r50"]:
        h = feat_hetero[name]
        print(f"  {name:20s} between_sd={h['between_sd']:8.3f}  max_within_sd={h['max_within_sd']:8.3f}  "
              f"ratio={h['ratio']:.2f}")

    # --- heterogeneidad intra vs entre escenario: coeficientes oraculo ---
    print("\n=== Heterogeneidad intra- vs entre-escenario: coeficientes oraculo (split-half bootstrap) ===")
    v0_global, e1_global, e2_global = dtg.calibrate_boresight(data["ioct_q"])
    stability = split_half_stability(data["ioct_q"], data["gt"], scenario, v0_global, e1_global, e2_global,
                                      n_trials=30, seed=args.seed)
    for i, name in enumerate(COEF_NAMES):
        print(f"  {name:8s} between_scenario_sd={stability['between_scenario_sd'][i]:10.3f}  "
              f"mean_split_half_diff={stability['mean_split_half_abs_diff'][i]:10.3f}  "
              f"ratio={stability['ratio_between_over_splithalf'][i]:.2f}")

    # --- correlaciones exploratorias (informativas, n=10, con caveat) ---
    print("\n=== Correlaciones exploratorias features de imagen (media por escenario) vs coeficiente oraculo (n=10 escenarios, INFORMATIVO, no usado para elegir el modelo final) ===")
    oracle_global = oracle_coefficients_per_scenario(data["ioct_q"], data["gt"], scenario,
                                                       v0_global, e1_global, e2_global)
    scens = sorted(set(scenario))
    X_scen_mean_global = np.stack([np.nanmean(X_img[scenario == s], axis=0) for s in scens], axis=0)
    Y_oracle_global = np.stack([oracle_global[s] for s in scens], axis=0)
    corr_table = {}
    for cname_i, cname in enumerate(COEF_NAMES):
        row = {}
        for fname_i, fname in enumerate(FEATURE_NAMES):
            r, p = sstats.pearsonr(X_scen_mean_global[:, fname_i], Y_oracle_global[:, cname_i])
            row[fname] = (float(r), float(p))
        corr_table[cname] = row
    for cname in ["s0", "c0_tx", "c0_ty"]:
        best = sorted(corr_table[cname].items(), key=lambda kv: -abs(kv[1][0]))[:3]
        print(f"  {cname}: top-3 |r| -> " + ", ".join(f"{f}(r={r:+.2f},p={p:.3f})" for f, (r, p) in best))

    # --- predictor lineal end-to-end, LOSO anidado ---
    print("\n=== Predictor de imagen -> coeficientes, LOSO anidado, AUC oficial end-to-end ===")
    image_loso = run_image_based_loso(data, X_img, PRIMARY_FEATURES)
    for scen, stats in sorted(image_loso["per_scenario"].items()):
        print(f"  scenario {scen}: n={stats['n']:4d} mean_err={stats['mean_error']:7.2f}px "
              f"median={stats['median_error']:7.2f}px AUC={stats['auc']:.4f}")
    print(f"POOLED (features={PRIMARY_FEATURES}): AUC={image_loso['pooled_auc']:.4f}  "
          f"mean_err={image_loso['pooled_mean_error']:.2f}px  median={image_loso['pooled_median_error']:.2f}px")

    # ablacion: solo intercepto + radio (2 params) -- ver si menos features generaliza mejor/peor
    image_loso_radius_only = run_image_based_loso(data, X_img, ["vis_r_equiv"])
    print(f"\nAblacion (solo vis_r_equiv, 2 params por objetivo): AUC={image_loso_radius_only['pooled_auc']:.4f} "
          f"mean_err={image_loso_radius_only['pooled_mean_error']:.2f}px")

    # ablacion: base ampliada con excentricidad y radio de perfil de brillo
    extra_features = PRIMARY_FEATURES + ["vis_eccentricity", "profile_edge_r50"]
    image_loso_extra = run_image_based_loso(data, X_img, extra_features)
    print(f"Ablacion (base ampliada, 6 params por objetivo): AUC={image_loso_extra['pooled_auc']:.4f} "
          f"mean_err={image_loso_extra['pooled_mean_error']:.2f}px")

    # ablacion: features de la mascara no-negra en vez de visibility.png (redundancia/robustez)
    nb_features = ["nb_r_equiv", "nb_centroid_y", "nb_centroid_x"]
    image_loso_nb = run_image_based_loso(data, X_img, nb_features)
    print(f"Ablacion (mascara no-negra en vez de visibility.png): AUC={image_loso_nb['pooled_auc']:.4f} "
          f"mean_err={image_loso_nb['pooled_mean_error']:.2f}px")

    # ablaciones de feature UNICA (2 params por objetivo, el minimo posible) --
    # las tres formas alternativas, todas a priori razonables, de medir "tamano
    # aparente del campo visible": mascara binaria (area, radio equivalente) vs
    # perfil radial de intensidad (mas robusto a oclusion por instrumentos).
    single_feature_results = {}
    for fname in ["vis_r_equiv", "vis_area_frac", "profile_edge_r50", "vis_bbox_cx"]:
        r = run_image_based_loso(data, X_img, [fname])
        single_feature_results[fname] = r
        print(f"Ablacion (solo {fname}, 2 params por objetivo): AUC={r['pooled_auc']:.4f} "
              f"mean_err={r['pooled_mean_error']:.2f}px")
    combo = run_image_based_loso(data, X_img, ["vis_r_equiv", "profile_edge_r50"])
    print(f"Ablacion (vis_r_equiv + profile_edge_r50, 3 params por objetivo): AUC={combo['pooled_auc']:.4f} "
          f"mean_err={combo['pooled_mean_error']:.2f}px")

    print("\n=== Chequeo nulo: feature de RUIDO GAUSSIANO en el mismo pipeline (1000 tiradas) ===")
    null_check = run_null_permutation_check(data, X_img, n_trials=1000, seed=7)
    print(f"AUC con ruido puro: media={null_check['auc_mean']:.4f} std={null_check['auc_std']:.4f} "
          f"min={null_check['auc_min']:.4f} max={null_check['auc_max']:.4f}  (n_trials={null_check['n_trials']})")
    print("Comparar contra esto, no contra 0, al juzgar si una feature real 'funciona'.")
    null_aucs = np.asarray(null_check["all_aucs"])
    print(f"Percentiles del nulo: p90={np.percentile(null_aucs,90):.4f}  p95={np.percentile(null_aucs,95):.4f}  "
          f"p99={np.percentile(null_aucs,99):.4f}")
    for fname, r in single_feature_results.items():
        pctl = float((null_aucs < r["pooled_auc"]).mean() * 100)
        z = float((r["pooled_auc"] - null_aucs.mean()) / null_aucs.std())
        print(f"  {fname:20s} AUC={r['pooled_auc']:.4f}  percentil_vs_nulo={pctl:5.1f}%  z={z:+.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = dict(
        n_cases=n,
        reference_global_auc=loso_global["pooled_auc"],
        reference_intercept_only_oracle_auc=intercept_oracle["pooled_auc"],
        reference_full_oracle_auc=full_oracle["pooled_auc"],
        feature_heterogeneity={k: v for k, v in feat_hetero.items()},
        coefficient_split_half_stability=stability,
        exploratory_correlations={
            cname: {fname: dict(r=r, p=p) for fname, (r, p) in row.items()}
            for cname, row in corr_table.items()
        },
        image_based_loso_primary=image_loso,
        image_based_loso_radius_only=image_loso_radius_only,
        image_based_loso_extra=image_loso_extra,
        image_based_loso_nb=image_loso_nb,
        single_feature_results={k: dict(pooled_auc=v["pooled_auc"], pooled_mean_error=v["pooled_mean_error"],
                                          per_scenario=v["per_scenario"])
                                 for k, v in single_feature_results.items()},
        combo_r_equiv_profile=dict(pooled_auc=combo["pooled_auc"], pooled_mean_error=combo["pooled_mean_error"],
                                     per_scenario=combo["per_scenario"]),
        null_permutation_check=null_check,
        primary_features=PRIMARY_FEATURES,
    )
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
