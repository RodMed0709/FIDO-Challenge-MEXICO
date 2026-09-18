#!/usr/bin/env python3
"""Genera las figuras del explicador de Task 2 a partir de un caso real.

Produce JPEGs listos para incrustar como data URI en un artifact:
  fig_fundus.jpg    la imagen de microscopio limpia
  fig_crosshair.jpg la misma con el cuadrado del escaneo iOCT dibujado desde el GT
  fig_enface.jpg    la proyeccion en-face del volumen (donde se ven los vasos)
  fig_overlay.jpg   el en-face deformado con la matriz real, encima del fundus

    python analysis/render_task2_figures.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_volume(volume_dir: Path) -> np.ndarray:
    slices = sorted(p for p in volume_dir.glob("*.png"))
    return np.stack([np.array(Image.open(p).convert("L")) for p in slices], axis=0)


def enface(volume: np.ndarray) -> np.ndarray:
    """Proyeccion frontal: promedio a lo largo del eje de profundidad.

    El volumen llega como (n_bscans, profundidad, ancho). Promediar sobre el eje
    1 aplana la profundidad y deja una vista de la retina "desde arriba", que es
    la que comparte estructura con el fundus.
    """
    projection = volume.mean(axis=1)
    projection -= projection.min()
    projection /= max(projection.max(), 1e-6)
    return (projection * 255).astype(np.uint8)


def save(image: Image.Image, path: Path, size: int = 640, quality: int = 82):
    image.convert("RGB").resize((size, size), Image.LANCZOS).save(
        path, "JPEG", quality=quality, optimize=True
    )
    kb = path.stat().st_size / 1024
    print(f"  {path.name:20s} {kb:7.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=PROJECT_ROOT / "data" / "_sample")
    parser.add_argument("--frame", default="00102")
    parser.add_argument("--scenario", default="Scenario_01")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    fundus = Image.open(args.sample / "Stereo Left" / args.frame / "microscope.png").convert("RGB")
    volume = load_volume(args.sample / "iOCT Microscope" / "Volume" / args.frame)
    annotation = json.loads(
        (PROJECT_ROOT / "data" / "_annotations" / "Task 2" / args.scenario /
         f"{args.frame}.json").read_text(encoding="utf-8")
    )
    matrix = np.asarray(annotation["Ground Truth"]["Task 2"], dtype=np.float64)

    print(f"fundus {fundus.size}   volumen {volume.shape}")
    print(f"M =\n{np.round(matrix, 2)}")

    # --- 1. fundus limpio ---
    save(fundus, args.out / "fig_fundus.jpg")

    # --- 2. fundus con el cuadrado del escaneo ---
    # Las esquinas del cuadrado unitario en coordenadas normalizadas [-1,1],
    # que es el espacio en el que vive el volumen iOCT.
    unit_corners = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    projected = (matrix @ unit_corners.T).T[:, :2]

    marked = fundus.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    outline = [tuple(p) for p in projected]
    draw.polygon(outline, fill=(63, 181, 190, 40))
    draw.line(outline + [outline[0]], fill=(63, 181, 190, 255), width=6)

    centre = matrix[:2, 2]
    arm = 26
    draw.line([(centre[0] - arm, centre[1]), (centre[0] + arm, centre[1])],
              fill=(255, 255, 255, 255), width=5)
    draw.line([(centre[0], centre[1] - arm), (centre[0], centre[1] + arm)],
              fill=(255, 255, 255, 255), width=5)
    for point in projected:
        draw.ellipse([point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9],
                     outline=(255, 255, 255, 255), width=4)
    save(marked, args.out / "fig_crosshair.jpg")

    # --- 3. proyeccion en-face ---
    projection = enface(volume)
    save(Image.fromarray(projection), args.out / "fig_enface.jpg")

    # --- 4. en-face deformado sobre el fundus ---
    # El en-face ocupa exactamente el cuadrado unitario, asi que basta mapear sus
    # esquinas a las esquinas proyectadas. PIL QUAD toma las esquinas del origen
    # en el orden (arriba-izq, abajo-izq, abajo-der, arriba-der) del destino.
    enface_rgb = Image.fromarray(projection).convert("RGB").resize(fundus.size, Image.LANCZOS)
    quad = (
        projected[0][0], projected[0][1],
        projected[3][0], projected[3][1],
        projected[2][0], projected[2][1],
        projected[1][0], projected[1][1],
    )
    warped = enface_rgb.transform(fundus.size, Image.QUAD, quad, Image.BICUBIC)

    # Recorta la mezcla al poligono para que el en-face no manche todo el fundus.
    mask = Image.new("L", fundus.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in projected], fill=150)
    overlay = fundus.copy()
    overlay.paste(Image.blend(fundus, warped, 0.75), (0, 0), mask)
    ImageDraw.Draw(overlay).line(outline + [outline[0]], fill=(63, 181, 190), width=6)
    save(overlay, args.out / "fig_overlay.jpg")

    # --- data URIs para incrustar ---
    manifest = {}
    for path in sorted(args.out.glob("fig_*.jpg")):
        manifest[path.stem] = "data:image/jpeg;base64," + base64.b64encode(
            path.read_bytes()).decode("ascii")
    (args.out / "figures_b64.json").write_text(json.dumps(manifest))
    total = sum(len(v) for v in manifest.values()) / 1024
    print(f"\n  data URIs -> figures_b64.json   ({total:.0f} KB en total)")


if __name__ == "__main__":
    main()
