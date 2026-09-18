#!/usr/bin/env python3
"""T1-R2 a escala completa: ¿se puede MEDIR la distancia cánula-ILM?

Usa las dos extracciones ya hechas por separado:
  data/_annotations/Task 1/<Escenario>/<frame>.json   -> GT + Surgical Tool
  data/_bscan_seg/Task 1/<Escenario>/<frame>/00.png,01.png -> máscaras B-scan

Aísla la CANNULA (T1-R1: es siempre la herramienta activa, 30/30 verificado;
el endoiluminador trae el centinela int32 max) para no contaminar la medición
si ambas herramientas caen en la clase InstrumentInOCT del B-scan.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENT_CLASS = 11  # InstrumentInOCT
ILM_CLASS = 1
SCALE = (4.0 / 512) * 1000  # = 7.8125, GROUND_TRUTH_DISTANCE_SCALE del scoring oficial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=ROOT / "data/_annotations/Task 1")
    p.add_argument("--masks", type=Path, default=ROOT / "data/_bscan_seg/Task 1")
    p.add_argument("--only-cannula", action="store_true", default=True)
    args = p.parse_args()

    pixels, gts, skipped, not_active = [], [], 0, 0

    for scen_dir in sorted(args.annotations.iterdir()):
        if not scen_dir.is_dir():
            continue
        mask_scen = args.masks / scen_dir.name
        if not mask_scen.is_dir():
            continue
        for json_path in scen_dir.glob("*.json"):
            frame_id = json_path.stem
            mask_dir = mask_scen / frame_id
            if not mask_dir.is_dir():
                continue
            data = json.loads(json_path.read_text(encoding="utf-8"))
            gt = data["Ground Truth"]["Task 1"][2] / SCALE

            if args.only_cannula:
                ilm = data.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
                if ilm is None or ilm > 1e6:
                    not_active += 1
                    continue

            for slice_path in sorted(mask_dir.glob("*.png")):
                mask = np.array(Image.open(slice_path))
                ys, xs = np.nonzero(mask == INSTRUMENT_CLASS)
                if ys.size == 0:
                    skipped += 1
                    continue
                deepest = np.argmax(ys)
                tip_row, tip_col = int(ys[deepest]), int(xs[deepest])

                low, high = max(0, tip_col - 5), min(mask.shape[1], tip_col + 6)
                ilm_ys, _ = np.nonzero(mask[:, low:high] == ILM_CLASS)
                if ilm_ys.size == 0:
                    skipped += 1
                    continue
                ilm_row = int(ilm_ys.min())

                pixels.append(ilm_row - tip_row)
                gts.append(gt)

    pixels = np.asarray(pixels, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)

    print(f"Frames con CANNULA activa: {len(list(args.annotations.rglob('*.json'))) - not_active}")
    print(f"  (saltados por no-cannula-activa: {not_active})")
    print(f"Mediciones válidas (ambos B-scans): {len(pixels)}  (saltadas por máscara vacía: {skipped})")

    if len(pixels) < 10:
        print("Muy pocos puntos.")
        return

    design = np.vstack([pixels, np.ones_like(pixels)]).T
    (a, b), *_ = np.linalg.lstsq(design, gts, rcond=None)
    pred = a * pixels + b
    ss_res = np.sum((gts - pred) ** 2)
    ss_tot = np.sum((gts - gts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\ndistancia_GT = {a:.4f} * pixel_gap + {b:.4f}")
    print(f"R² = {r2:.4f}")
    print(f"pixel_gap: [{pixels.min():.1f}, {pixels.max():.1f}]  gt: [{gts.min():.1f}, {gts.max():.1f}]")


if __name__ == "__main__":
    main()
