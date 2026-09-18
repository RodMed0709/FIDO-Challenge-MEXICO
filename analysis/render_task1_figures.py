#!/usr/bin/env python3
"""Genera las figuras del explicador de Task 1 a partir de un caso real.

Produce JPEGs listos para incrustar como data URI en un artifact:
  t1_fig_fundus.jpg    la imagen de microscopio limpia
  t1_fig_keypoint.jpg  la misma con el keypoint GT (punta de la cánula) marcado
  t1_fig_bscan_raw.jpg uno de los 2 B-scans ortogonales, crudo
  t1_fig_bscan_seg.jpg el mismo B-scan con la segmentación coloreada + la
                        distancia medida (Ilm <-> punta del instrumento)

    python analysis/render_task1_figures.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# vendor/fido/Dataset Explorer/constants.py::GenericLabels — no se copia a mano,
# pero para esta figura basta con los valores ya verificados en este proyecto.
ILM_CLASS = 1
RPE_CLASS = 2
VESSEL_CLASS = 3
INSTRUMENT_CLASS = 11
SHADOW_CLASS = 13  # no documentada en el enum oficial, ver verify_task1_distance_geometry.py

CLASS_COLORS = {
    0: (18, 18, 24),        # fondo
    1: (255, 209, 102),     # Ilm
    2: (6, 214, 160),       # Rpe
    3: (239, 71, 111),      # ArteriesOrVeins
    11: (17, 138, 178),     # InstrumentInOCT (cánula)
    13: (90, 90, 100),      # sombra no documentada
}
DEFAULT_COLOR = (60, 60, 68)


def colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value in np.unique(mask):
        rgb[mask == value] = CLASS_COLORS.get(int(value), DEFAULT_COLOR)
    return Image.fromarray(rgb)


def save(image: Image.Image, path: Path, size: int = 640, quality: int = 85):
    image.convert("RGB").resize((size, size), Image.LANCZOS).save(
        path, "JPEG", quality=quality, optimize=True
    )
    kb = path.stat().st_size / 1024
    print(f"  {path.name:22s} {kb:7.1f} KB")


def tip_and_ilm(mask: np.ndarray):
    ys, xs = np.nonzero(mask == INSTRUMENT_CLASS)
    if ys.size == 0:
        return None
    deepest = np.argmax(ys)
    tip_row, tip_col = int(ys[deepest]), int(xs[deepest])
    window = 5
    low, high = max(0, tip_col - window), min(mask.shape[1], tip_col + window + 1)
    strip = mask[:, low:high] == ILM_CLASS
    ilm_ys, _ = np.nonzero(strip)
    ilm_row = int(ilm_ys.min()) if ilm_ys.size else None
    return tip_row, tip_col, ilm_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                         default=PROJECT_ROOT / "data" / "Mock Test" / "Task 1")
    parser.add_argument("--scenario", default="Scenario_11")
    parser.add_argument("--frame", default="04990")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenario_dir = args.root / args.scenario
    frame = args.frame

    fundus = Image.open(scenario_dir / "Stereo Left" / frame / "microscope.png").convert("RGB")
    gt = json.loads((scenario_dir / "Numerical" / f"{frame}.json").read_text(encoding="utf-8"))
    keypoint = gt["Ground Truth"]["Task 1"][:2]
    distance_gt = gt["Ground Truth"]["Task 1"][2] / 10.0

    print(f"fundus {fundus.size}   keypoint (cánula) = ({keypoint[0]:.1f}, {keypoint[1]:.1f})")
    print(f"distancia GT herramienta-tejido = {distance_gt:.2f} px")

    # --- 1. fundus limpio ---
    save(fundus, args.out / "t1_fig_fundus.jpg")

    # --- 2. fundus con el keypoint marcado ---
    marked = fundus.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    x, y = keypoint
    r = 22
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 82, 82, 255), width=6)
    arm = 34
    draw.line([(x - arm, y), (x + arm, y)], fill=(255, 82, 82, 255), width=3)
    draw.line([(x, y - arm), (x, y + arm)], fill=(255, 82, 82, 255), width=3)
    save(marked, args.out / "t1_fig_keypoint.jpg")

    # --- 3 y 4. B-scan crudo + segmentación coloreada ---
    bscan_dir = scenario_dir / "iOCT Microscope" / "Bscan" / frame
    bscan = np.array(Image.open(bscan_dir / "00.png").convert("L"))
    save(Image.fromarray(bscan).convert("RGB"), args.out / "t1_fig_bscan_raw.jpg")

    seg = np.array(Image.open(bscan_dir / "Segmentation" / "00.png"))
    seg_rgb = colorize(seg).convert("RGBA")
    draw = ImageDraw.Draw(seg_rgb, "RGBA")

    found = tip_and_ilm(seg)
    if found is not None:
        tip_row, tip_col, ilm_row = found
        draw.ellipse([tip_col - 8, tip_row - 8, tip_col + 8, tip_row + 8],
                     outline=(255, 255, 255, 255), width=3)
        if ilm_row is not None:
            draw.line([(tip_col, tip_row), (tip_col, ilm_row)],
                       fill=(255, 255, 255, 255), width=3)
            pixel_gap = ilm_row - tip_row
            measured = 0.9370 * pixel_gap + 3.4277  # ajuste de T1-R2 reescalado x1.28 (10/7.8125), R²=0.9916
            draw.text((tip_col + 14, (tip_row + ilm_row) // 2 - 8),
                       f"{pixel_gap}px -> ~{measured:.0f} (GT={distance_gt:.0f})",
                       fill=(255, 255, 255, 255))
    save(seg_rgb, args.out / "t1_fig_bscan_seg.jpg")

    # --- data URIs ---
    manifest_path = args.out / "figures_b64.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for path in sorted(args.out.glob("t1_fig_*.jpg")):
        manifest[path.stem] = "data:image/jpeg;base64," + base64.b64encode(
            path.read_bytes()).decode("ascii")
    manifest_path.write_text(json.dumps(manifest))
    total = sum(len(v) for v in manifest.values()) / 1024
    print(f"\n  data URIs -> {manifest_path}   ({total:.0f} KB en total)")


if __name__ == "__main__":
    main()
