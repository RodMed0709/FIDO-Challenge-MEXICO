#!/usr/bin/env python3
"""¿Se puede MEDIR la distancia herramienta-tejido en vez de regresarla?

Peldaño T1-R2 de ATTACK_LADDER.md. La literatura del grupo CAMP (mismo grupo
que organiza el challenge) mide, no regresa: segmentan la punta del
instrumento y la ILM en el B-scan y toman la separación axial. El challenge
regala esas máscaras como supervisión auxiliar (InstrumentInOCT, Ilm, Rpe).

Un intento anterior, manual, sobre 5 frames del Mock Test NO cuadró (ver
ATTACK_LADDER.md, historial). Este script lo reintenta con el método correcto:
ajuste por mínimos cuadrados sobre muchos casos, localizando la punta en la
columna A-scan correcta, no una comparación caso por caso a ojo.

Requiere los PNG de segmentación de los B-scans, que están en el volumen de
RunPod (raw/Task1/*.zip), no localmente salvo en el Mock Test. Primero corre
sobre el Mock Test (5 casos, ya en local) como prueba de humo del método; si
hace falta más señal, se corre en el pod contra el dataset completo.

    python analysis/verify_task1_distance_geometry.py
    python analysis/verify_task1_distance_geometry.py --root "data/Task 1_extracted"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "Mock Test" / "Task 1"

# Clases del enum GenericLabels en vendor/fido/Dataset Explorer/constants.py
INSTRUMENT_CLASS = 11   # InstrumentInOCT
ILM_CLASS = 1            # Ilm
# Clase 13 observada en los datos pero no documentada en el enum oficial —
# aparece como banda vertical hasta el borde inferior, compatible con la
# sombra del instrumento. Se prueba por separado, nunca mezclada con la 11.
UNDOCUMENTED_SHADOW_CLASS = 13

GROUND_TRUTH_DISTANCE_SCALE = (4.0 / 512) * 1000  # = 7.8125, igual que en scoring_keypoints.py del bundle


def find_cases(root: Path):
    """(scenario_dir, frame_id) para cada frame con Numerical + Bscan disponibles."""
    cases = []
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        numerical = scenario_dir / "Numerical"
        bscan_root = scenario_dir / "iOCT Microscope" / "Bscan"
        if not numerical.is_dir() or not bscan_root.is_dir():
            continue
        for json_path in sorted(numerical.glob("*.json")):
            frame_id = json_path.stem
            if (bscan_root / frame_id).is_dir():
                cases.append((scenario_dir, frame_id))
    return cases


def tip_and_ilm_column(mask: np.ndarray, instrument_class: int):
    """Punta = píxel de la clase de instrumento con mayor fila (z más profundo,
    más cerca de la retina). Devuelve (fila_punta, columna) o None si no hay
    píxeles de esa clase."""
    ys, xs = np.nonzero(mask == instrument_class)
    if ys.size == 0:
        return None
    deepest = np.argmax(ys)
    return int(ys[deepest]), int(xs[deepest])


def ilm_row_near_column(mask: np.ndarray, column: int, window: int = 5):
    """Fila de la ILM más superficial (menor z) cerca de la columna dada.

    La ILM es la capa más externa de la retina; su borde superior es la
    superficie que un instrumento alcanza primero al acercarse desde el vítreo.
    """
    low = max(0, column - window)
    high = min(mask.shape[1], column + window + 1)
    strip = mask[:, low:high] == ILM_CLASS
    ys, _ = np.nonzero(strip)
    if ys.size == 0:
        return None
    return int(ys.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--annotations", type=Path, default=None,
                        help="Carpeta con los Numerical/*.json ya extraídos "
                             "(si difiere de --root, p.ej. data/_annotations/Task 1)")
    parser.add_argument("--instrument-class", type=int, default=INSTRUMENT_CLASS,
                        help=f"Clase a tratar como instrumento (default {INSTRUMENT_CLASS} "
                             f"= InstrumentInOCT; probar también {UNDOCUMENTED_SHADOW_CLASS})")
    parser.add_argument("--only-cannula", action="store_true",
                        help="Saltar frames donde CANNULA no es la herramienta activa "
                             "(T1-R1: la distancia GT es siempre de la cánula, nunca del "
                             "endoiluminador — evita contaminar la medición si ambas caen "
                             "en la misma clase de segmentación del B-scan)")
    args = parser.parse_args()

    cases = find_cases(args.root)
    if not cases:
        raise SystemExit(f"No hay casos con B-scan en {args.root}")

    pixels, ground_truth, skipped, not_cannula_active = [], [], 0, 0

    for scenario_dir, frame_id in cases:
        annotations_dir = args.annotations or scenario_dir
        gt_path = (annotations_dir / scenario_dir.name / "Numerical" / f"{frame_id}.json"
                  if args.annotations else scenario_dir / "Numerical" / f"{frame_id}.json")
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_distance = gt_data["Ground Truth"]["Task 1"][2] / GROUND_TRUTH_DISTANCE_SCALE

        if args.only_cannula:
            cannula_ilm = gt_data.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
            if cannula_ilm is None or cannula_ilm > 1e6:
                not_cannula_active += 1
                continue

        bscan_dir = scenario_dir / "iOCT Microscope" / "Bscan" / frame_id
        for slice_name in ("00", "01"):
            seg_path = bscan_dir / "Segmentation" / f"{slice_name}.png"
            if not seg_path.exists():
                continue
            mask = np.array(Image.open(seg_path))

            tip = tip_and_ilm_column(mask, args.instrument_class)
            if tip is None:
                skipped += 1
                continue
            tip_row, tip_col = tip
            ilm_row = ilm_row_near_column(mask, tip_col)
            if ilm_row is None:
                skipped += 1
                continue

            pixel_gap = ilm_row - tip_row  # positivo si el instrumento está por encima de la ILM
            pixels.append(pixel_gap)
            ground_truth.append(gt_distance)

    pixels = np.asarray(pixels, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    print(f"Clase de instrumento probada: {args.instrument_class}")
    print(f"Casos con B-scan disponibles : {len(cases)}")
    print(f"Mediciones válidas           : {len(pixels)}")
    print(f"Saltados (sin instrumento/ILM detectable en la máscara): {skipped}")

    if len(pixels) < 3:
        print("\nMuy pocos puntos para ajustar. El método no es aplicable así, o "
              "la clase de instrumento probada no es la correcta.")
        return

    # Ajuste por mínimos cuadrados: distancia_GT = a*pixel_gap + b
    design = np.vstack([pixels, np.ones_like(pixels)]).T
    (a, b), residuals, rank, singular_values = np.linalg.lstsq(design, ground_truth, rcond=None)
    predicted = a * pixels + b
    ss_res = np.sum((ground_truth - predicted) ** 2)
    ss_tot = np.sum((ground_truth - ground_truth.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nAjuste lineal: distancia_GT = {a:.4f} * pixel_gap + {b:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"Rango de pixel_gap observado: [{pixels.min():.1f}, {pixels.max():.1f}]")
    print(f"Rango de distancia_GT       : [{ground_truth.min():.1f}, {ground_truth.max():.1f}]")

    print("\nInterpretación:")
    if r_squared > 0.95 and abs(b) < 0.1 * ground_truth.std():
        print("  R² alto y b≈0 -> la distancia es la separación axial pura a la ILM.")
        print("  MEDIR es viable: reemplaza la regresión ciega por segmentación + geometría.")
    elif r_squared > 0.7:
        print("  R² razonable pero con offset o ruido -> hay señal geométrica real,")
        print("  probablemente con una referencia desplazada (capa virtual entre ILM y RPE)")
        print("  o error de segmentación. Vale la pena una versión con corrección residual")
        print("  aprendida, no descartar el enfoque geométrico.")
    else:
        print("  R² bajo -> la relación no es axial simple en un B-scan. Probar con el")
        print("  segundo B-scan ortogonal y/o norma euclídea combinando ambos antes de")
        print("  rendirse. Si sigue bajo, usar regresión distribucional (DFL) en su lugar.")


if __name__ == "__main__":
    main()
