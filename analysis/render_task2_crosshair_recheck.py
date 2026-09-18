"""Renders para inspección visual de R2 revisitado (ver
`experiments/97-t2-crosshair-recheck/INFORME.md`).

Para un puñado de casos (train + mock), guarda:
  - <case>_full.png            fundus tal cual llega a `opmi_image`
  - <case>_overlay.png         mismo fundus + crosshair anotado (rojo) +
                                rectángulo escaneado vía project_corners (cian)
  - <case>_crop.png            recorte 300x300 centrado en el crosshair
  - <case>_crop_stretch.png    mismo recorte con ecualización de histograma
                                (CLAHE) para forzar visible cualquier
                                artefacto sutil
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fido.geometry import project_corners  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "97-t2-crosshair-recheck" / "renders"
CROP_HALF = 150


def crosshair_segments(data: dict) -> list[np.ndarray]:
    ch = data.get("Keypoints", {}).get("iOCT Microscope Crosshair", {})
    segs = []
    for a, b in (("Start 0", "End 0"), ("Start 1", "End 1")):
        pa, pb = ch.get(a), ch.get(b)
        if pa is not None and pb is not None:
            segs.append(np.array([pa, pb], dtype=np.float64))
    return segs


def render_case(json_path: Path, img_path: Path, out_prefix: str) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [skip] no se pudo leer {img_path}")
        return
    matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
    segs = crosshair_segments(data)
    center = matrix[:2, 2]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / f"{out_prefix}_full.png"), img)

    overlay = img.copy()
    corners = project_corners(matrix)
    poly = np.round(corners).astype(np.int32)
    cv2.polylines(overlay, [poly], isClosed=True, color=(255, 255, 0), thickness=2)  # cian (BGR)
    for seg in segs:
        p0 = tuple(np.round(seg[0]).astype(int))
        p1 = tuple(np.round(seg[1]).astype(int))
        cv2.line(overlay, p0, p1, color=(0, 0, 255), thickness=2)  # rojo
    cv2.circle(overlay, tuple(np.round(center).astype(int)), 5, (0, 255, 0), -1)  # verde
    cv2.imwrite(str(OUT_DIR / f"{out_prefix}_overlay.png"), overlay)

    h, w = img.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, x1 = max(0, cx - CROP_HALF), min(w, cx + CROP_HALF)
    y0, y1 = max(0, cy - CROP_HALF), min(h, cy + CROP_HALF)
    crop = img[y0:y1, x0:x1].copy()
    crop_overlay = crop.copy()
    for seg in segs:
        p0 = tuple(np.round(seg[0] - [x0, y0]).astype(int))
        p1 = tuple(np.round(seg[1] - [x0, y0]).astype(int))
        cv2.line(crop_overlay, p0, p1, color=(0, 0, 255), thickness=1)
    cv2.imwrite(str(OUT_DIR / f"{out_prefix}_crop.png"),
                cv2.resize(crop_overlay, (crop.shape[1] * 3, crop.shape[0] * 3),
                           interpolation=cv2.INTER_NEAREST))

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    stretched = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    for seg in segs:
        p0 = tuple(np.round(seg[0] - [x0, y0]).astype(int))
        p1 = tuple(np.round(seg[1] - [x0, y0]).astype(int))
        cv2.line(stretched, p0, p1, color=(0, 0, 255), thickness=1)
    cv2.imwrite(str(OUT_DIR / f"{out_prefix}_crop_stretch.png"),
                cv2.resize(stretched, (stretched.shape[1] * 3, stretched.shape[0] * 3),
                           interpolation=cv2.INTER_NEAREST))
    print(f"  ok {out_prefix}")


def main() -> None:
    cases = [
        ("train_s03_00000", REPO_ROOT / "data" / "_annotations" / "Task 2" / "Scenario_03" / "00000.json",
         REPO_ROOT / "data" / "Task 2" / "Scenario_03" / "Stereo Left" / "00000" / "microscope.png"),
        ("train_s03_00040", REPO_ROOT / "data" / "_annotations" / "Task 2" / "Scenario_03" / "00040.json",
         REPO_ROOT / "data" / "Task 2" / "Scenario_03" / "Stereo Left" / "00040" / "microscope.png"),
        ("train_s03_00082", REPO_ROOT / "data" / "_annotations" / "Task 2" / "Scenario_03" / "00082.json",
         REPO_ROOT / "data" / "Task 2" / "Scenario_03" / "Stereo Left" / "00082" / "microscope.png"),
    ]
    cache = REPO_ROOT / "experiments" / "97-t2-crosshair-recheck" / "cache"
    for scenario_dir in sorted(cache.glob("Scenario_*")):
        jsons = sorted((scenario_dir / "Numerical").glob("*.json"))
        if not jsons:
            continue
        for j in (jsons[0], jsons[len(jsons) // 2], jsons[-1]):
            fid = j.stem
            img = scenario_dir / "Stereo Left" / fid / "microscope.png"
            if img.is_file():
                cases.append((f"train_{scenario_dir.name}_{fid}", j, img))

    mock_dir = REPO_ROOT / "data" / "Mock Test" / "Task 2" / "Scenario_12"
    for j in sorted((mock_dir / "Numerical").glob("*.json")):
        fid = j.stem
        img = mock_dir / "Stereo Left" / fid / "microscope.png"
        if img.is_file():
            cases.append((f"mock_s12_{fid}", j, img))

    print(f"Renderizando {len(cases)} casos en {OUT_DIR}")
    for prefix, json_path, img_path in cases:
        if json_path.is_file() and img_path.is_file():
            render_case(json_path, img_path, prefix)
        else:
            print(f"  [skip] falta archivo para {prefix}")


if __name__ == "__main__":
    main()
