#!/usr/bin/env python3
"""Experimento 102 — búsqueda densa por solapamiento de máscaras vasculares
(4 DOF: tx, ty, theta, s) para Task 2.

Motivación (ver el encargo original / ATTACK_LADDER.md T2-R4): el
leaderboard correlaciona score con tiempo de inferencia — los que puntúan
>0.24 tardan >267s/100 casos (4.5-6 s/caso), consistente con una búsqueda
densa de pose, no una sola pasada de red. T2-R4 (Motor 1,
`src/fido/models/vessel_template_match.py`) ya probó una versión de esto
(NCC vía `cv2.matchTemplate` sobre la DENSIDAD continua de vaso) y midió
"oracle NCC" ~ruido sobre solo 4 casos válidos del Mock Test — pero esa
medición es un test de ENRIQUECIMIENTO (¿la señal es fuerte en la pose
correcta?), no un test de ARGMAX (¿la pose correcta es el máximo de la
función de puntuación?). Este script mide lo segundo, con máscaras
BINARIAS (no densidad continua), 4 funciones de puntuación (NCC, IoU, Dice,
distance-transform/chamfer), sobre >=100 casos reales de varios escenarios.

Reusa `_build_patch` de `vessel_template_match.py` (T2-R4) sin modificarlo:
misma convención geométrica ya verificada por
`analysis/verify_vessel_template_match_synthetic.py` (error ~0.09px en el
caso sintético autoconsistente). Convención NATIVA (sin canonicalizar): GT
tal cual viene del JSON, `reflect=True` fijo (T2-R1/T2-R3, verificado en
1214 casos de entrenamiento).

ADVERTENCIA (autocrítica obligatoria, ver encargo): usa máscaras GT de
vasos en AMBOS lados (fundus y en-face del OCT). Esto mide el TECHO del
método (¿existe la señal, es la pose correcta el argmax?), NO el rendimiento
desplegable (que dependería de un segmentador entrenado en cada lado). No
confundir los dos números.

Uso:
    python analysis/dense_search_task2_vessels.py --out experiments/102-t2-dense-search/results.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fido.data.common import (  # noqa: E402
    enface_vessel_density,
    load_label_map,
    load_volume_label_maps,
)
from fido.geometry import (  # noqa: E402
    compose_similarity,
    corner_error,
    decompose_similarity,
    fit_closed_form_similarity,
    project_corners,
)
from fido.models.vessel_template_match import _REF_CORNERS_REFLECTED, _build_patch  # noqa: E402

METHODS = ("ncc", "iou", "dice", "chamfer", "chamfer_sym")
# Métodos donde el score decrece con la distancia (queremos minimizar la
# distancia, no maximizar el score crudo) -- se niegan antes de comparar,
# de forma que en TODOS los métodos "más alto es mejor".
MIN_PATCH_VESSEL_PIXELS = 8
EPS = 1e-6


# --------------------------------------------------------------------------
# Descubrimiento de casos (independiente de `find_task2_cases`: nuestra
# extracción parcial de zips sólo trae `Volume/<frame>/Segmentation/*.png`,
# no los slices RAW en escala de grises que `find_task2_cases` usa para
# detectar `has_volume`).
# --------------------------------------------------------------------------

def find_cases(root: Path, scenario_filter: str | None = None) -> list[dict]:
    cases = []
    for scenario_dir in sorted(root.glob("Scenario_*")):
        if scenario_filter and scenario_dir.name != scenario_filter:
            continue
        numerical_dir = scenario_dir / "Numerical"
        if not numerical_dir.is_dir():
            continue
        for json_path in sorted(numerical_dir.glob("*.json")):
            frame_id = json_path.stem
            vessel_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
            fundus_path = scenario_dir / "Stereo Left" / frame_id / "microscope.png"
            volume_seg_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id / "Segmentation"
            if not (vessel_path.exists() and fundus_path.exists() and volume_seg_dir.is_dir()):
                continue
            if not any(volume_seg_dir.glob("*.png")):
                continue
            cases.append({
                "scenario": scenario_dir.name,
                "frame_id": frame_id,
                "scenario_dir": str(scenario_dir),
                "json_path": str(json_path),
                "root": str(root),
            })
    return cases


# --------------------------------------------------------------------------
# Construcción de máscaras
# --------------------------------------------------------------------------

def load_fundus_vessel_mask(scenario_dir: Path, frame_id: str) -> np.ndarray:
    path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
    return (load_label_map(path) > 0).astype(np.uint8)


def load_enface_vessel_mask(scenario_dir: Path, frame_id: str) -> np.ndarray:
    volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
    seg_volume = load_volume_label_maps(volume_dir / "Segmentation")
    density = enface_vessel_density(seg_volume, vessel_class=3)
    return (density > 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Núcleo de puntuación: dado un patch binario ya construido, calcula las 5
# superficies de puntuación de una sola pasada (todas derivadas de 2-3
# llamadas a cv2.matchTemplate, todas O(H*W log) vía DFT interno de OpenCV
# -- no hay bucle Python sobre traslaciones).
# --------------------------------------------------------------------------

def build_fundus_products(fundus_bin: np.ndarray) -> dict:
    fundus_f32 = fundus_bin.astype(np.float32)
    fundus_u8 = (fundus_bin * 255).astype(np.uint8)
    # distanceTransform(fondo) = distancia en px de cada pixel al vaso GT
    # de fundus MAS CERCANO. DIST_L2 + mask 5x5 = aproximación subpíxel
    # estándar de OpenCV (error < 2% frente a la distancia euclídea exacta).
    fundus_dist = cv2.distanceTransform(255 - fundus_u8, cv2.DIST_L2, 5).astype(np.float32)
    return {"bin": fundus_f32, "dist": fundus_dist}


def score_surfaces(fundus_products: dict, patch_bin: np.ndarray) -> dict | None:
    patch_f = patch_bin.astype(np.float32)
    patch_sum = float(patch_f.sum())
    if patch_sum < MIN_PATCH_VESSEL_PIXELS:
        return None

    fundus_bin = fundus_products["bin"]
    fundus_dist = fundus_products["dist"]

    ncc = cv2.matchTemplate(fundus_bin, patch_f, cv2.TM_CCOEFF_NORMED)

    intersection = cv2.matchTemplate(fundus_bin, patch_f, cv2.TM_CCORR)
    ones = np.ones_like(patch_f)
    local_fundus_sum = cv2.matchTemplate(fundus_bin, ones, cv2.TM_CCORR)
    union = local_fundus_sum + patch_sum - intersection
    iou = intersection / np.maximum(union, EPS)
    dice = (2.0 * intersection) / np.maximum(local_fundus_sum + patch_sum, EPS)

    # Chamfer asimétrico: distancia media de los píxeles de vaso del patch
    # (proyección OCT) al vaso GT de fundus más cercano. Menor = mejor ->
    # se niega para que "más alto = mejor" en todos los métodos.
    dist_sum_dir1 = cv2.matchTemplate(fundus_dist, patch_f, cv2.TM_CCORR)
    chamfer = -(dist_sum_dir1 / patch_sum)

    # Chamfer simétrico: promedia con la dirección inversa (vaso GT de
    # fundus -> vaso OCT del patch más cercano). `patch_dist` se computa
    # UNA vez por celda (scale, angle) -- barato frente al costo de
    # construir el patch en sí.
    patch_u8 = (patch_bin * 255).astype(np.uint8)
    patch_dist = cv2.distanceTransform(255 - patch_u8, cv2.DIST_L2, 5).astype(np.float32)
    # `dist_sum_dir2`/`local_fundus_sum` son matemáticamente >=0 (correlación
    # de dos arrays no-negativos), pero `matchTemplate` internamente usa DFT
    # (FFT) -- el ruido de redondeo puede dar valores ligeramente negativos.
    # Sin el clip a 0, un `local_fundus_sum` de ruido (~1e-7) diluido por ese
    # ruido negativo dispara la división a magnitudes absurdas (visto en el
    # smoke test: chamfer_sym positivo de cientos, imposible para un
    # promedio de distancias negado). Umbral real (no solo > 0): una ventana
    # necesita AL MENOS unos pocos píxeles de vaso GT de fundus para que el
    # promedio dirección-2 tenga sentido físico.
    MIN_LOCAL_FUNDUS_PIXELS = 3.0
    dist_sum_dir2 = np.maximum(cv2.matchTemplate(fundus_bin, patch_dist, cv2.TM_CCORR), 0.0)
    local_fundus_sum_safe = np.maximum(local_fundus_sum, MIN_LOCAL_FUNDUS_PIXELS)
    dir2_avg = dist_sum_dir2 / local_fundus_sum_safe
    # Donde la ventana de fundus no tiene suficiente vaso GT, dir2 no está
    # definida de forma confiable -- se cae de vuelta al chamfer asimétrico
    # puro en esas celdas en vez de inventar un valor con denominador ínfimo.
    dir2_avg = np.where(local_fundus_sum >= MIN_LOCAL_FUNDUS_PIXELS, dir2_avg, -chamfer)
    chamfer_sym = -0.5 * ((dist_sum_dir1 / patch_sum) + dir2_avg)

    return {"ncc": ncc, "iou": iou, "dice": dice, "chamfer": chamfer, "chamfer_sym": chamfer_sym}


# --------------------------------------------------------------------------
# Búsqueda densa gruesa + refinamiento fino
# --------------------------------------------------------------------------

def _corners_from_window(corners_local: np.ndarray, argmax_idx: tuple[int, int]) -> np.ndarray:
    window_xy = np.array([argmax_idx[1], argmax_idx[0]], dtype=np.float64)  # (x,y) top-left
    return corners_local + window_xy


def coarse_dense_search(fundus_products: dict, enface_bin: np.ndarray,
                         scale_min: float, scale_max: float, scale_step: float,
                         angle_step_deg: float):
    scales = np.arange(scale_min, scale_max + 1e-6, scale_step)
    angles_deg = np.arange(0.0, 360.0, angle_step_deg)

    best = {m: None for m in METHODS}
    cell_maxima = {m: [] for m in METHODS}
    n_cells_evaluated = 0

    for scale in scales:
        for angle_deg in angles_deg:
            angle_rad = np.radians(angle_deg)
            patch_bin, corners_local = _build_patch(enface_bin.astype(np.float32), float(scale), angle_rad)
            ph, pw = patch_bin.shape
            h_f, w_f = fundus_products["bin"].shape
            if ph < 4 or pw < 4 or ph > h_f or pw > w_f:
                continue
            surfaces = score_surfaces(fundus_products, patch_bin)
            if surfaces is None:
                continue
            n_cells_evaluated += 1
            for method, surface in surfaces.items():
                idx = np.unravel_index(np.argmax(surface), surface.shape)
                val = float(surface[idx])
                cell_maxima[method].append(val)
                if best[method] is None or val > best[method]["score"]:
                    pred_corners = _corners_from_window(corners_local, idx)
                    best[method] = {
                        "score": val, "scale": float(scale), "angle_deg": float(angle_deg),
                        "pred_corners": pred_corners.tolist(),
                    }
    return best, cell_maxima, n_cells_evaluated


def refine_pose(fundus_products: dict, enface_bin: np.ndarray, method: str, coarse_best: dict,
                 scale_window: float, angle_window_deg: float, n_steps: int = 8):
    scale0, angle0 = coarse_best["scale"], coarse_best["angle_deg"]
    scales = np.linspace(scale0 - scale_window, scale0 + scale_window, n_steps)
    angles = np.linspace(angle0 - angle_window_deg, angle0 + angle_window_deg, n_steps)
    best = dict(coarse_best)
    h_f, w_f = fundus_products["bin"].shape
    for scale in scales:
        if scale <= 0:
            continue
        for angle_deg in angles:
            patch_bin, corners_local = _build_patch(enface_bin.astype(np.float32), float(scale), np.radians(angle_deg))
            ph, pw = patch_bin.shape
            if ph < 4 or pw < 4 or ph > h_f or pw > w_f:
                continue
            surfaces = score_surfaces(fundus_products, patch_bin)
            if surfaces is None:
                continue
            surface = surfaces[method]
            idx = np.unravel_index(np.argmax(surface), surface.shape)
            val = float(surface[idx])
            if val > best["score"]:
                pred_corners = _corners_from_window(corners_local, idx)
                best = {"score": val, "scale": float(scale), "angle_deg": float(angle_deg),
                        "pred_corners": pred_corners.tolist()}
    return best


def evaluate_gt_pose(fundus_products: dict, enface_bin: np.ndarray, gt_matrix: np.ndarray):
    """Puntúa la alineación EXACTA del GT (sin búsqueda), redondeando la
    ventana a la posición entera más cercana. Ninguna libertad de argmax --
    esto es lo que un oráculo perfecto vería en la pose correcta."""
    params = decompose_similarity(gt_matrix, reflect=True)
    scale = float(params["scale"])
    cos_t, sin_t = float(params["cos_theta"]), float(params["sin_theta"])
    angle_rad = float(np.arctan2(sin_t, cos_t))
    patch_bin, corners_local = _build_patch(enface_bin.astype(np.float32), scale, angle_rad)
    h_f, w_f = fundus_products["bin"].shape
    ph, pw = patch_bin.shape
    if ph < 4 or pw < 4 or ph > h_f or pw > w_f:
        return None
    surfaces = score_surfaces(fundus_products, patch_bin)
    if surfaces is None:
        return None

    gt_corners = project_corners(gt_matrix)  # (4,2), posición real en fundus
    window_xy = np.mean(gt_corners - corners_local, axis=0)
    x, y = int(round(window_xy[0])), int(round(window_xy[1]))
    out = {}
    for method, surface in surfaces.items():
        sh, sw = surface.shape
        if not (0 <= y < sh and 0 <= x < sw):
            out[method] = None
        else:
            out[method] = float(surface[y, x])
    return out


# --------------------------------------------------------------------------
# Un caso completo
# --------------------------------------------------------------------------

def run_case(case: dict, scale_min: float, scale_max: float, scale_step: float,
             angle_step_deg: float, refine: bool) -> dict:
    t0 = time.time()
    scenario_dir = Path(case["scenario_dir"])
    frame_id = case["frame_id"]
    record = {"scenario": case["scenario"], "frame_id": frame_id}
    try:
        fundus_bin = load_fundus_vessel_mask(scenario_dir, frame_id)
        enface_bin = load_enface_vessel_mask(scenario_dir, frame_id)
        record["fundus_vessel_frac"] = float(fundus_bin.mean())
        record["enface_vessel_frac"] = float(enface_bin.mean())

        data = json.loads(Path(case["json_path"]).read_text(encoding="utf-8"))
        gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)
        gt_scale = float(decompose_similarity(gt_matrix, reflect=True)["scale"])
        record["gt_scale"] = gt_scale

        fundus_products = build_fundus_products(fundus_bin)

        coarse_best, cell_maxima, n_cells = coarse_dense_search(
            fundus_products, enface_bin, scale_min, scale_max, scale_step, angle_step_deg,
        )
        record["n_cells_evaluated"] = n_cells

        gt_scores = evaluate_gt_pose(fundus_products, enface_bin, gt_matrix)
        record["gt_scores"] = gt_scores

        for method in METHODS:
            cb = coarse_best[method]
            if cb is None:
                record[method] = {"status": "no_valid_cell"}
                continue
            final = cb
            if refine:
                final = refine_pose(
                    fundus_products, enface_bin, method, cb,
                    scale_window=scale_step, angle_window_deg=angle_step_deg,
                )
            pred_corners = np.array(final["pred_corners"], dtype=np.float64)
            fit = fit_closed_form_similarity(pred_corners, ref_corners=_REF_CORNERS_REFLECTED)
            pred_matrix = compose_similarity(
                fit["tx"], fit["ty"], fit["cos_theta"], fit["sin_theta"], fit["scale"], reflect=True,
            )
            err = float(corner_error(pred_matrix, gt_matrix))

            maxima = np.asarray(cell_maxima[method], dtype=np.float64)
            gt_score = gt_scores.get(method) if gt_scores else None
            percentile = None
            if gt_score is not None and maxima.size > 0:
                percentile = float(np.mean(maxima <= gt_score) * 100.0)

            record[method] = {
                "status": "ok",
                "coarse_scale": cb["scale"], "coarse_angle_deg": cb["angle_deg"], "coarse_score": cb["score"],
                "final_scale": final["scale"], "final_angle_deg": final["angle_deg"], "final_score": final["score"],
                "corner_error": err,
                "gt_score": gt_score,
                "gt_score_percentile_vs_cell_maxima": percentile,
                "n_cell_maxima": int(maxima.size),
            }
        record["elapsed_s"] = time.time() - t0
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{exc}\n{traceback.format_exc()}"
        record["elapsed_s"] = time.time() - t0
    return record


def _run_case_star(args):
    return run_case(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/Task 2"])
    ap.add_argument("--scenario", default=None, help="filtra a un solo escenario, p.ej. Scenario_07")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-min", type=float, default=126.0)
    ap.add_argument("--scale-max", type=float, default=230.0)
    ap.add_argument("--scale-step", type=float, default=8.0)
    ap.add_argument("--angle-step-deg", type=float, default=10.0)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=None, help="limita el numero de casos (debug)")
    args = ap.parse_args()

    all_cases = []
    for root in args.roots:
        all_cases.extend(find_cases(PROJECT_ROOT / root, scenario_filter=args.scenario))
    if args.limit:
        all_cases = all_cases[: args.limit]
    print(f"{len(all_cases)} casos encontrados en {args.roots}")

    tasks = [
        (case, args.scale_min, args.scale_max, args.scale_step, args.angle_step_deg, not args.no_refine)
        for case in all_cases
    ]

    results = []
    t_start = time.time()
    if args.workers <= 1:
        for i, t in enumerate(tasks):
            results.append(_run_case_star(t))
            print(f"[{i+1}/{len(tasks)}] {results[-1]['scenario']}/{results[-1]['frame_id']} "
                  f"status={results[-1]['status']} elapsed={results[-1].get('elapsed_s', -1):.1f}s")
    else:
        with mp.Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(_run_case_star, tasks)):
                results.append(rec)
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    print(f"[{i+1}/{len(tasks)}] acumulado {time.time()-t_start:.0f}s")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEscrito {out_path} ({len(results)} casos, {time.time()-t_start:.0f}s total)")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    print(f"ok={n_ok} error={n_err}")
    if n_err:
        for r in results:
            if r["status"] == "error":
                print(f"  ERROR {r['scenario']}/{r['frame_id']}: {r['error'].splitlines()[-1]}")


if __name__ == "__main__":
    main()
