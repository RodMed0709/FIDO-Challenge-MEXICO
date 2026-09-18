#!/usr/bin/env python3
"""T1-86: decompone el efecto de la corrección de profundidad sobre distance_auc.

Pregunta: de la subida esperada de `distance_auc`, ¿cuánto viene de quitar el
sesgo del 28% (constantes reescaladas) y cuánto de que `MAX_THRESHOLD_DIST`
pasó de 10 a 20? Se responde con el oráculo geométrico (máscaras GT, no un
checkpoint entrenado) sobre los 61,691 frames locales en
`data/_annotations/Task 1` + `data/_bscan_seg/Task 1`, usando la función REAL
`auc_from_errors` vendorizada -- no una reimplementación.

Es un TECHO (segmentación perfecta), no una medición del checkpoint real de
r04/r05. Sirve para aislar los dos efectos pedidos en el peldaño T1-86 sobre
datos reales a escala completa, no sobre los 5 casos del Mock Test.

    python analysis/measure_t1_distance_rescale_effect.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENT_CLASS = 11  # InstrumentInOCT
ILM_CLASS = 1
OLD_SCALE = 10.0
NEW_SCALE = (4.0 / 512) * 1000  # 7.8125, GROUND_TRUTH_DISTANCE_SCALE oficial corregido
K = OLD_SCALE / NEW_SCALE  # 1.28 exacto

DISTANCE_FALLBACK_PX = 159.2  # constante ya usada en las submissions, sin tocar

# auc_from_errors REAL del scorer vendorizado, no una reimplementación
spec = importlib.util.spec_from_file_location(
    "scoring_keypoints",
    ROOT / "vendor" / "fido" / "Codabench Bundle" / "scoring_program" / "scoring_keypoints.py",
)
scoring_keypoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring_keypoints)
auc_from_errors = scoring_keypoints.auc_from_errors
assert scoring_keypoints.MAX_THRESHOLD_DIST == 20
assert abs(scoring_keypoints.GROUND_TRUTH_DISTANCE_SCALE - NEW_SCALE) < 1e-9


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1,
                        help="Usa 1 de cada N frames por escenario (submuestreo "
                             "sistemático, no aleatorio, para acelerar sobre I/O "
                             "lento). 1 = dataset completo.")
    args = parser.parse_args()

    annotations_root = ROOT / "data" / "_annotations" / "Task 1"
    masks_root = ROOT / "data" / "_bscan_seg" / "Task 1"

    covered_gap, covered_gt_stored = [], []
    uncovered_gt_stored = []
    total_frames, not_cannula_active = 0, 0
    started = time.time()

    for scen_dir in sorted(annotations_root.iterdir()):
        if not scen_dir.is_dir():
            continue
        mask_scen = masks_root / scen_dir.name
        if not mask_scen.is_dir():
            continue
        json_paths = sorted(scen_dir.glob("*.json"))[::args.stride]
        print(f"[{time.time()-started:6.1f}s] {scen_dir.name}: {len(json_paths):,} frames "
              f"(stride={args.stride})", flush=True)
        for json_path in json_paths:
            frame_id = json_path.stem
            mask_dir = mask_scen / frame_id
            if not mask_dir.is_dir():
                continue
            total_frames += 1
            data = json.loads(json_path.read_text(encoding="utf-8"))
            stored = float(data["Ground Truth"]["Task 1"][2])

            ilm = data.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
            if ilm is None or ilm > 1e6:
                not_cannula_active += 1
                continue

            frame_gap = None
            for slice_path in sorted(mask_dir.glob("*.png")):
                mask = np.array(Image.open(slice_path))
                ys, xs = np.nonzero(mask == INSTRUMENT_CLASS)
                if ys.size == 0:
                    continue
                deepest = np.argmax(ys)
                tip_row, tip_col = int(ys[deepest]), int(xs[deepest])
                low, high = max(0, tip_col - 5), min(mask.shape[1], tip_col + 6)
                ilm_ys, _ = np.nonzero(mask[:, low:high] == ILM_CLASS)
                if ilm_ys.size == 0:
                    continue
                gap = float(int(ilm_ys.min()) - tip_row)
                frame_gap = gap if frame_gap is None else (frame_gap + gap) / 2.0

            if frame_gap is None:
                uncovered_gt_stored.append(stored)
                continue
            covered_gap.append(frame_gap)
            covered_gt_stored.append(stored)

    covered_gap = np.asarray(covered_gap, dtype=np.float64)
    covered_gt_stored = np.asarray(covered_gt_stored, dtype=np.float64)
    uncovered_gt_stored = np.asarray(uncovered_gt_stored, dtype=np.float64)

    n_covered = len(covered_gap)
    n_uncovered = len(uncovered_gt_stored)
    n_active = n_covered + n_uncovered
    coverage = n_covered / n_active if n_active else float("nan")

    print(f"Frames con anotación+máscara: {total_frames:,}")
    print(f"  saltados por cannula inactiva: {not_cannula_active:,}")
    print(f"  cannula activa, SIN medición geométrica en ningún B-scan: {n_uncovered:,}")
    print(f"  cannula activa, CON medición geométrica: {n_covered:,}")
    print(f"  coverage sobre frames con cannula activa: {coverage:.4f}")

    target_old_covered = covered_gt_stored / OLD_SCALE
    target_new_covered = covered_gt_stored / NEW_SCALE
    target_old_uncovered = uncovered_gt_stored / OLD_SCALE
    target_new_uncovered = uncovered_gt_stored / NEW_SCALE

    def fit(x, y):
        design = np.vstack([x, np.ones_like(x)]).T
        (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = a * x + b
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return a, b, r2, pred

    a_old, b_old, r2_old, pred_old_covered = fit(covered_gap, target_old_covered)
    a_new, b_new, r2_new, pred_new_covered = fit(covered_gap, target_new_covered)

    print(f"\nAjuste OLS target VIEJO (GT/10):      a={a_old:.4f} b={b_old:.4f} R2={r2_old:.4f}"
          f"  n={n_covered:,}")
    print(f"Ajuste OLS target NUEVO (GT/7.8125):   a={a_new:.4f} b={b_new:.4f} R2={r2_new:.4f}"
          f"  n={n_covered:,}")
    print(f"Chequeo reescalado exacto (K={K}): a_old*K={a_old*K:.4f} vs a_new={a_new:.4f}"
          f"  |  b_old*K={b_old*K:.4f} vs b_new={b_new:.4f}")

    err_old_covered = np.abs(pred_old_covered - target_old_covered)
    err_new_covered = np.abs(pred_new_covered - target_new_covered)
    print(f"Chequeo: err_new/err_old media (debe ser {K} exacto) = "
          f"{float((err_new_covered / np.where(err_old_covered == 0, np.nan, err_old_covered)).mean()):.6f}")

    print(f"\nResiduo del ajuste NUEVO (px, target={{'GT/7.8125'}}):"
          f" media={err_new_covered.mean():.3f} mediana={np.median(err_new_covered):.3f}"
          f" p90={np.percentile(err_new_covered, 90):.3f} p99={np.percentile(err_new_covered, 99):.3f}")

    # --- 2x2: aisla sesgo (target viejo/nuevo) x umbral (10/20), SOLO casos cubiertos ---
    auc_A = auc_from_errors(list(err_old_covered), max_threshold=10)  # antes (sesgo + umbral viejo)
    auc_B = auc_from_errors(list(err_old_covered), max_threshold=20)  # solo umbral ensanchado
    auc_C = auc_from_errors(list(err_new_covered), max_threshold=10)  # solo sesgo corregido
    auc_D = auc_from_errors(list(err_new_covered), max_threshold=20)  # las dos correcciones (real)

    print(f"\nAUC del oraculo, SOLO casos cubiertos geometricamente (n={n_covered:,}, "
          f"coverage={coverage:.1%} de los frames con cannula activa):")
    print(f"  A) target viejo, umbral 10 (antes de ambas correcciones) = {auc_A:.4f}")
    print(f"  B) target viejo, umbral 20 (solo ensancha el umbral)     = {auc_B:.4f}")
    print(f"  C) target nuevo, umbral 10 (solo quita el sesgo)         = {auc_C:.4f}")
    print(f"  D) target nuevo, umbral 20 (las dos correcciones, real)  = {auc_D:.4f}")
    print(f"  Delta total D-A = {auc_D - auc_A:+.4f}"
          f"  |  atribuible a sesgo (C-A) = {auc_C - auc_A:+.4f}"
          f"  |  atribuible a umbral (B-A) = {auc_B - auc_A:+.4f}"
          f"  |  interaccion (D-C-B+A) = {(auc_D - auc_C - auc_B + auc_A):+.4f}")

    # --- Extiende a TODOS los frames con cannula activa: fallback fijo para
    #     los no cubiertos, igual que hace inference.py en produccion.
    fb_old = np.full(n_uncovered, DISTANCE_FALLBACK_PX)
    err_old_uncovered = np.abs(fb_old - target_old_uncovered)
    err_new_uncovered = np.abs(fb_old - target_new_uncovered)  # mismo fallback, target nuevo

    def all_case_auc(err_covered, err_uncovered, threshold):
        all_errors = np.concatenate([err_covered, err_uncovered])
        return auc_from_errors(list(all_errors), max_threshold=threshold)

    all_A = all_case_auc(err_old_covered, err_old_uncovered, 10)
    all_B = all_case_auc(err_old_covered, err_old_uncovered, 20)
    all_C = all_case_auc(err_new_covered, err_new_uncovered, 10)
    all_D = all_case_auc(err_new_covered, err_new_uncovered, 20)

    print(f"\nAUC del oraculo sobre TODOS los frames con cannula activa "
          f"(n={n_active:,}, fallback={DISTANCE_FALLBACK_PX}px SIN reescalar para los no cubiertos):")
    print(f"  A) target viejo, umbral 10 (antes de ambas correcciones) = {all_A:.4f}")
    print(f"  B) target viejo, umbral 20 (solo ensancha el umbral)     = {all_B:.4f}")
    print(f"  C) target nuevo, umbral 10 (solo quita el sesgo)         = {all_C:.4f}")
    print(f"  D) target nuevo, umbral 20 (las dos correcciones, real)  = {all_D:.4f}")
    print(f"  Delta total D-A = {all_D - all_A:+.4f}"
          f"  |  atribuible a sesgo (C-A) = {all_C - all_A:+.4f}"
          f"  |  atribuible a umbral (B-A) = {all_B - all_A:+.4f}"
          f"  |  interaccion (D-C-B+A) = {(all_D - all_C - all_B + all_A):+.4f}")


if __name__ == "__main__":
    main()
