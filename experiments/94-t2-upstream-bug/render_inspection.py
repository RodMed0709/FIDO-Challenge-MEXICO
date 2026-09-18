"""Render en-face projection strategies vs. fundus for visual inspection (T2-94).

Standalone: reimplements the small pieces of measure_task2_oracle_signal.py /
verify_task2_enface_convention.py needed (warp_fundus, projection_images,
convention_matrix) instead of importing them, so it can run against the
extracted case folders without pulling in the full oracle CLI / multiprocessing
machinery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.data.common import (  # noqa: E402
    enface_projection,
    canonicalize_task2_enface,
    load_volume_slices,
    load_volume_label_maps,
)

VESSEL_CLASS = 3
INSTRUMENT_CLASSES = (8, 10, 11)

EXTRACTED = Path(__file__).resolve().parent / "extracted"
OUT_DIR = Path(__file__).resolve().parent / "renders"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = tuple(
    (f"{'transpose' if transpose else 'identity'}__"
     f"{'flip_u' if flip_u else 'keep_u'}__{'flip_v' if flip_v else 'keep_v'}",
     transpose, flip_u, flip_v)
    for transpose in (False, True)
    for flip_u in (False, True)
    for flip_v in (False, True)
)
CONVENTIONS = {name: (t, fu, fv) for name, t, fu, fv in VARIANTS}


def transform_uv(uv, transpose, flip_u, flip_v):
    transformed = uv.copy()
    if flip_u:
        transformed[:, 0] = 1.0 - transformed[:, 0]
    if flip_v:
        transformed[:, 1] = 1.0 - transformed[:, 1]
    return transformed[:, ::-1] if transpose else transformed


def convention_matrix(convention: str) -> np.ndarray:
    transpose, flip_u, flip_v = CONVENTIONS[convention]
    basis = transform_uv(np.array([[0., 0.], [1., 0.], [0., 1.]]), transpose, flip_u, flip_v)
    return np.array([
        [basis[1, 0] - basis[0, 0], basis[2, 0] - basis[0, 0], basis[0, 0]],
        [basis[1, 1] - basis[0, 1], basis[2, 1] - basis[0, 1], basis[0, 1]],
        [0., 0., 1.],
    ])


def warp_fundus(gray: np.ndarray, matrix: np.ndarray, shape: tuple[int, int], convention: str) -> np.ndarray:
    h, w = shape
    uv_to_pixel = np.array([[1.0 / (w - 1), 0, 0], [0, 1.0 / (h - 1), 0], [0, 0, 1]])
    return cv2.warpPerspective(
        gray, matrix @ np.linalg.inv(convention_matrix(convention)) @ uv_to_pixel, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )


def project_corners(matrix: np.ndarray) -> np.ndarray:
    ref = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
    projected = (matrix @ ref.T).T
    return projected[:, :2] / projected[:, 2:3]


def to_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def label_panel(img: np.ndarray, text: str, target_h: int = 260) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = target_h / h
    img = cv2.resize(img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h + 30, img.shape[1], 3), dtype=np.uint8)
    canvas[:target_h] = img
    cv2.putText(canvas, text, (4, target_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def hstack_pad(panels: list[np.ndarray], pad: int = 6) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            extra = np.zeros((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, extra])
        padded.append(p)
        padded.append(np.zeros((h, pad, 3), dtype=np.uint8))
    return np.hstack(padded[:-1])


def process_case(scenario: str, frame: str, convention: str = "transpose__flip_u__flip_v") -> dict:
    case_dir = EXTRACTED / scenario
    volume_dir = case_dir / "iOCT Microscope" / "Volume" / frame
    seg_dir = volume_dir / "Segmentation"
    fundus_path = case_dir / "Stereo Left" / frame / "microscope.png"
    vessel_seg_path = case_dir / "Stereo Left" / frame / "Segmentation" / "arteriesorveins.png"
    json_path = REPO_ROOT / "data" / "_annotations" / "Task 2" / scenario / f"{frame}.json"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)

    volume = load_volume_slices(volume_dir)  # (128, depth, width) native
    seg = load_volume_label_maps(seg_dir) if seg_dir.is_dir() else None

    fundus_rgb = np.array(Image.open(fundus_path).convert("RGB"))
    fundus_gray = cv2.cvtColor(fundus_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    fundus_vessel = np.array(Image.open(vessel_seg_path)) if vessel_seg_path.exists() else None

    volume_f = volume.astype(np.float32) / 255.0
    projections = {
        "mean_full_depth (pipeline actual)": volume_f.mean(axis=1),
        "MIP_max_full_depth": volume_f.max(axis=1),
        "min_full_depth": volume_f.min(axis=1),
    }
    n_bands = 8
    for i, slab in enumerate(np.array_split(volume_f, n_bands, axis=1)):
        projections[f"slab_{i}_of_{n_bands}"] = slab.mean(axis=1)
    if seg is not None:
        vessel_any = np.any(seg == VESSEL_CLASS, axis=1).astype(np.float32)
        projections["OCT_vessel_class_ANY_depth (ground truth)"] = vessel_any
        instrument_any = np.any(np.isin(seg, INSTRUMENT_CLASSES), axis=1).astype(np.float32)
        projections["OCT_instrument_class_ANY_depth (ground truth)"] = instrument_any

    # aligned fundus patches: one per projection shape/orientation.
    warped = {}
    for name, proj in projections.items():
        warped[name] = warp_fundus(fundus_gray, matrix, proj.shape, convention)
    if fundus_vessel is not None:
        vessel_bin = (fundus_vessel > 0).astype(np.float32)
        # use same shape as OCT vessel-class projection for a direct mask-vs-mask compare
        target_shape = projections.get("OCT_vessel_class_ANY_depth (ground truth)", projections["mean_full_depth (pipeline actual)"]).shape
        warped["fundus_vessel_mask_warped"] = warp_fundus(vessel_bin, matrix, target_shape, convention)

    # full fundus with OCT footprint corners drawn (native+ "correct" convention corners are same:
    # the matrix's own unit square corners always map to the same 4 fundus points regardless of
    # which array-axis convention we pick for *sampling* -- convention only changes how we index
    # the projection array, not where the footprint sits on the fundus.)
    corners = project_corners(matrix).astype(np.int32)
    fundus_annot = fundus_rgb.copy()
    cv2.polylines(fundus_annot, [corners.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0), thickness=3)
    for i, c in enumerate(corners):
        cv2.circle(fundus_annot, tuple(c), 6, (0, 255, 0), -1)
        cv2.putText(fundus_annot, str(i), tuple(c + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    case_id = f"{scenario}_{frame}"
    out_path = OUT_DIR / f"{case_id}.png"

    rows = []
    # Row 1: full fundus with footprint box + close-up crop
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    x0, y0 = max(0, x0 - 20), max(0, y0 - 20)
    x1, y1 = min(fundus_rgb.shape[1], x1 + 20), min(fundus_rgb.shape[0], y1 + 20)
    crop = fundus_rgb[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else fundus_rgb
    row1 = hstack_pad([
        label_panel(cv2.cvtColor(fundus_annot, cv2.COLOR_RGB2BGR), f"{case_id}: fundus + OCT footprint (0-1-2-3 = corner order)", target_h=380),
        label_panel(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), "zoom on footprint region", target_h=380),
    ])
    rows.append(row1)

    # Row 2: aligned fundus patch vs mean/MIP/min projections
    row2 = hstack_pad([
        label_panel(to_u8(warped["mean_full_depth (pipeline actual)"]), "fundus warped to OCT grid (aligned)"),
        label_panel(to_u8(projections["mean_full_depth (pipeline actual)"]), "enface MEAN full depth (= pipeline today)"),
        label_panel(to_u8(projections["MIP_max_full_depth"]), "enface MAX (MIP) full depth"),
        label_panel(to_u8(projections["min_full_depth"]), "enface MIN full depth"),
    ])
    rows.append(row2)

    # Row 3: depth slabs
    slab_panels = [label_panel(to_u8(warped["mean_full_depth (pipeline actual)"]), "fundus (ref)")]
    for i in range(n_bands):
        key = f"slab_{i}_of_{n_bands}"
        slab_panels.append(label_panel(to_u8(projections[key]), f"slab {i}/{n_bands} mean-depth"))
    rows.append(hstack_pad(slab_panels))

    # Row 4: vessel-class ground truth (OCT) vs fundus vessel mask, both warped/aligned
    if seg is not None:
        panels4 = [
            label_panel(to_u8(warped["mean_full_depth (pipeline actual)"]), "fundus warped (ref)"),
            label_panel(to_u8(projections["OCT_vessel_class_ANY_depth (ground truth)"]), "OCT seg: vessel class, ANY depth"),
        ]
        if "fundus_vessel_mask_warped" in warped:
            panels4.append(label_panel(to_u8(warped["fundus_vessel_mask_warped"]), "fundus seg: arteriesorveins, warped to OCT grid"))
        panels4.append(label_panel(to_u8(projections["OCT_instrument_class_ANY_depth (ground truth)"]), "OCT seg: instrument classes, ANY depth"))
        rows.append(hstack_pad(panels4))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            extra = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, extra])
        padded_rows.append(r)
        padded_rows.append(np.zeros((8, max_w, 3), dtype=np.uint8))
    canvas = np.vstack(padded_rows[:-1])
    cv2.imwrite(str(out_path), canvas)
    print(f"wrote {out_path} shape={canvas.shape}")

    # quick numeric summary (NCC) for the console / report
    def ncc(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 32:
            return float("nan")
        av, bv = a[valid].astype(np.float64), b[valid].astype(np.float64)
        av = av - av.mean()
        bv = bv - bv.mean()
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        return float(np.dot(av, bv) / denom) if denom > 0 else float("nan")

    summary = {"case_id": case_id}
    for name, proj in projections.items():
        summary[f"ncc_{name}"] = ncc(warped[name], proj)
    return summary


if __name__ == "__main__":
    cases = [
        ("Scenario_01", "00000"),
        ("Scenario_01", "00030"),
        ("Scenario_01", "00075"),
        ("Scenario_03", "00000"),
        ("Scenario_03", "00040"),
    ]
    summaries = []
    for scenario, frame in cases:
        try:
            summaries.append(process_case(scenario, frame))
        except Exception as exc:
            print(f"FAILED {scenario}/{frame}: {exc}")
    for s in summaries:
        print(s)
