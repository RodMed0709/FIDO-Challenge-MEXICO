"""Measure oracle Task 2 cross-modal signal without training a model.

Each case uses ``seed + case_index`` for its null RNG, so reports are identical
regardless of the number of worker processes or their completion order.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.data.common import load_rgb, load_volume_label_maps, load_volume_slices  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402
from fido.geometry import project_corners  # noqa: E402
from verify_task2_enface_convention import VARIANTS, transform_uv  # noqa: E402

VESSEL_CLASS = 3
INSTRUMENT_CLASSES = (8, 10, 11)  # Forceps, Endoilluminator, InstrumentInOCT
PIXEL_RADII = (0, 2, 5, 10, 20)
FRACTION_RADII = (0.005, 0.01, 0.02, 0.04)
NULL_SAMPLES = 200

CONVENTIONS = {
    name: (transpose, flip_u, flip_v)
    for name, transpose, flip_u, flip_v in VARIANTS
}
CONVENTION_ALIASES = {
    name.replace("__", "_").replace("flip_u", "flipu").replace("keep_u", "keepu")
        .replace("flip_v", "flipv").replace("keep_v", "keepv"): name
    for name in CONVENTIONS
}
DEFAULT_CONVENTION = "identity__keep_u__keep_v"


def map_fundus_to_uv(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points))])
    uv_h = (np.linalg.inv(matrix) @ points_h.T).T
    return uv_h[:, :2] / uv_h[:, 2:3]


def convention_matrix(convention: str) -> np.ndarray:
    """Return the homogeneous uv-to-en-face transform for a named convention."""
    transpose, flip_u, flip_v = CONVENTIONS[convention]
    basis = transform_uv(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                         transpose, flip_u, flip_v)
    return np.array([
        [basis[1, 0] - basis[0, 0], basis[2, 0] - basis[0, 0], basis[0, 0]],
        [basis[1, 1] - basis[0, 1], basis[2, 1] - basis[0, 1], basis[0, 1]],
        [0., 0., 1.],
    ])


def uv_to_pixels(uv: np.ndarray, shape: tuple[int, int], convention: str) -> np.ndarray:
    h, w = shape
    uv_h = np.column_stack([uv, np.ones(len(uv))])
    enface_uv = (convention_matrix(convention) @ uv_h.T).T[:, :2]
    return enface_uv * np.array([w - 1, h - 1], dtype=np.float64)


def points_from_group(data: dict, group: str, names: tuple[str, ...] | None = None) -> np.ndarray:
    values = data.get("Keypoints", {}).get(group, {})
    keys = names if names is not None else tuple(k for k in values if k.startswith("V_"))
    points = [values.get(key) for key in keys if values.get(key) is not None]
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def inside_uv(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    uv = map_fundus_to_uv(points, matrix)
    return uv[np.all((uv >= 0.0) & (uv <= 1.0), axis=1)]


def hit_rates(uv: np.ndarray, mask: np.ndarray, convention: str,
              radii: tuple[int, ...] | tuple[float, ...], radii_mode: str) -> dict[float, float]:
    if not len(uv):
        return {radius: np.nan for radius in radii}
    h, w = mask.shape
    sampling = (1.0 / (h - 1), 1.0 / (w - 1)) if radii_mode == "fraction" else None
    distance = distance_transform_edt(~mask.astype(bool), sampling=sampling)
    pixels = uv_to_pixels(uv, mask.shape, convention)
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, mask.shape[0] - 1)
    sampled = distance[y, x]
    return {radius: float(np.mean(sampled <= radius)) for radius in radii}


def projection_images(volume: np.ndarray, seg: np.ndarray | None) -> dict[str, np.ndarray]:
    volume_f = volume.astype(np.float32) / 255.0
    projections = {
        "mean": volume_f.mean(axis=1),
        "MIP": volume_f.max(axis=1),
        "min": volume_f.min(axis=1),
    }
    for index, slab in enumerate(np.array_split(volume_f, 4, axis=1)):
        projections[f"slab_{index}"] = slab.mean(axis=1)
    gy, gx = np.gradient(volume_f, axis=(1, 2))
    projections["gradient"] = np.sqrt(gx * gx + gy * gy).mean(axis=1)
    if seg is not None:
        projections["vessel_density"] = (seg == VESSEL_CLASS).mean(axis=1).astype(np.float32)
    return projections


def warp_fundus(gray: np.ndarray, matrix: np.ndarray, shape: tuple[int, int],
                convention: str) -> np.ndarray:
    h, w = shape
    uv_to_pixel = np.array([[1.0 / (w - 1), 0, 0], [0, 1.0 / (h - 1), 0], [0, 0, 1]])
    return cv2.warpPerspective(
        gray, matrix @ np.linalg.inv(convention_matrix(convention)) @ uv_to_pixel, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
    )


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 32:
        return np.nan
    av, bv = a[valid].astype(np.float64), b[valid].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else np.nan


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 32:
        return np.nan
    hist, _, _ = np.histogram2d(a[valid], b[valid], bins=32)
    probability = hist / hist.sum()
    px = probability.sum(axis=1, keepdims=True)
    py = probability.sum(axis=0, keepdims=True)
    expected = px @ py
    nz = probability > 0
    return float(np.sum(probability[nz] * np.log(probability[nz] / expected[nz])))


def random_translations(matrix: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    offsets = project_corners(matrix) - matrix[:2, 2]
    low = -offsets.min(axis=0)
    high = np.array([1023.0, 1023.0]) - offsets.max(axis=0)
    if np.any(low > high):
        low, high = np.zeros(2), np.full(2, 1023.0)
    return rng.uniform(low, high, size=(count, 2))


def crosshair_error(data: dict, matrix: np.ndarray) -> float:
    crosshair = data.get("Keypoints", {}).get("iOCT Microscope Crosshair", {})
    expected = {
        "Start 1": matrix[:2, 2] - matrix[:2, 0],
        "End 1": matrix[:2, 2] + matrix[:2, 0],
        "Start 0": matrix[:2, 2] + matrix[:2, 1],
    }
    errors = [np.linalg.norm(np.asarray(crosshair[name]) - point)
              for name, point in expected.items() if crosshair.get(name) is not None]
    if len(errors) != 3:
        raise ValueError("crosshair self-test requires Start 0, Start 1, and End 1")
    return float(max(errors))


def aggregate_rates(rows: list[dict[float, float]], radii: tuple) -> dict[float, float]:
    return {radius: float(np.nanmean([row[radius] for row in rows])) for radius in radii}


def fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.4f}"


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def finite_std(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.std()) if len(finite) else np.nan


def measure_case(case: dict, case_index: int, seed: int, convention: str,
                 radii: tuple, radii_mode: str) -> dict:
    """Load and measure one case, returning either metrics or a skip reason."""
    case_id = f"{case['scenario']}/{case['frame_id']}"
    result = {"case_id": case_id, "loaded": False, "error": None}
    try:
        rng = np.random.default_rng(seed + case_index)
        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
        volume_dir = case["scenario_dir"] / "iOCT Microscope" / "Volume" / case["frame_id"]
        volume = load_volume_slices(volume_dir)
        fundus = load_rgb(case["scenario_dir"] / "Stereo Left" / case["frame_id"] / "microscope.png")
        gray = cv2.cvtColor(fundus, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        seg_dir = volume_dir / "Segmentation"
        seg = load_volume_label_maps(seg_dir) if seg_dir.is_dir() and any(seg_dir.glob("*.png")) else None
        result["loaded"] = True
        result["self_test_error"] = crosshair_error(data, matrix)

        b_values = {}
        projections = projection_images(volume, seg)
        translations = random_translations(matrix, rng, NULL_SAMPLES)
        for name, projection in projections.items():
            gt_patch = warp_fundus(gray, matrix, projection.shape, convention)
            gt_ncc, gt_mi = ncc(gt_patch, projection), mutual_information(gt_patch, projection)
            null_ncc, null_mi = [], []
            for translation in translations:
                null_matrix = matrix.copy()
                null_matrix[:2, 2] = translation
                patch = warp_fundus(gray, null_matrix, projection.shape, convention)
                null_ncc.append(ncc(patch, projection))
                null_mi.append(mutual_information(patch, projection))
            null_ncc = np.asarray(null_ncc)
            null_mi = np.asarray(null_mi)
            ncc_std = finite_std(null_ncc)
            null_ncc_mean = finite_mean(null_ncc)
            zscore = (gt_ncc - null_ncc_mean) / ncc_std if ncc_std > 0 else np.nan
            b_values[name] = (gt_ncc, zscore, gt_mi, finite_mean(null_mi))
        result["b_values"] = b_values
        result["seg_missing"] = seg is None
        if seg is None:
            return result

        vessel_mask = np.any(seg == VESSEL_CLASS, axis=1)
        vessel_uv = inside_uv(points_from_group(data, "Vasculature"), matrix)
        result["a_count"] = len(vessel_uv)
        result["a_observed"] = hit_rates(vessel_uv, vessel_mask, convention, radii, radii_mode) if len(vessel_uv) else None
        result["a_null"] = hit_rates(rng.random((len(vessel_uv), 2)), vessel_mask, convention, radii, radii_mode) if len(vessel_uv) else None

        instrument_points = np.concatenate([
            points_from_group(data, "Endgripping Forceps",
                              ("Right Head Tip", "Left Head Tip", "Joint Tip", "Start")),
            points_from_group(data, "Endoilluminator", ("Tip", "Start")),
        ], axis=0)
        instrument_uv = inside_uv(instrument_points, matrix)
        result["c_count"] = len(instrument_uv)
        result["instrument_inside"] = bool(len(instrument_uv))
        if len(instrument_uv):
            instrument_mask = np.any(np.isin(seg, INSTRUMENT_CLASSES), axis=1)
            result["c_observed"] = hit_rates(instrument_uv, instrument_mask, convention, radii, radii_mode)
            result["c_null"] = hit_rates(rng.random((len(instrument_uv), 2)), instrument_mask, convention, radii, radii_mode)
        else:
            result["c_observed"] = result["c_null"] = None
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/Mock Test/Task 2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--enface-convention", choices=tuple(CONVENTIONS) + tuple(CONVENTION_ALIASES),
                        default=DEFAULT_CONVENTION)
    parser.add_argument("--radii-mode", choices=("pixels", "fraction"), default="pixels")
    args = parser.parse_args()
    args.enface_convention = CONVENTION_ALIASES.get(args.enface_convention, args.enface_convention)
    radii = FRACTION_RADII if args.radii_mode == "fraction" else PIXEL_RADII
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    cases = find_task2_cases(args.root)
    if args.limit is not None:
        cases = cases[:args.limit]

    a_observed, a_null, a_counts = [], [], []
    c_observed, c_null, c_counts = [], [], []
    instrument_cases_inside = 0
    b_values: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    loaded = skipped = seg_missing = 0
    self_test_error = None
    warnings_log: list[str] = []

    started = time.monotonic()
    results: list[dict | None] = [None] * len(cases)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(measure_case, case, index, args.seed, args.enface_convention,
                                   radii, args.radii_mode): index
                   for index, case in enumerate(cases)}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                case = cases[index]
                result = {"case_id": f"{case['scenario']}/{case['frame_id']}", "loaded": False,
                          "error": f"worker process failure: {exc}"}
            results[index] = result
            status = "ok" if result["error"] is None else f"skipped: {result['error']}"
            print(f"[{completed}/{len(cases)}] {result['case_id']} {status}", flush=True)
            if completed % 25 == 0:
                elapsed = time.monotonic() - started
                rate = completed / elapsed * 60 if elapsed > 0 else np.nan
                eta = (len(cases) - completed) / rate if rate > 0 else np.nan
                print(f"[{completed}/{len(cases)}] {rate:.1f} casos/min, ETA {eta:.1f} min", flush=True)

    for result in results:
        assert result is not None
        case_id = result["case_id"]
        loaded += int(result["loaded"])
        if result["error"] is not None:
            skipped += 1
            message = f"{case_id}: skipped after load/measurement failure: {result['error']}"
            warnings.warn(message)
            warnings_log.append(message)
            continue
        if self_test_error is None:
            self_test_error = result["self_test_error"]
            if self_test_error >= 1e-2:
                skipped += 1
                message = f"{case_id}: skipped after load/measurement failure: crosshair convention " \
                          f"error {self_test_error:.6f}px >= 0.01px"
                warnings.warn(message)
                warnings_log.append(message)
                continue
        for name, values in result["b_values"].items():
            b_values[name].append(values)
        if result["seg_missing"]:
            seg_missing += 1
            message = f"{case_id}: no volume Segmentation; omitted from A/C, retained in B"
            warnings.warn(message)
            warnings_log.append(message)
            continue
        a_counts.append(result["a_count"])
        if result["a_observed"] is not None:
            a_observed.append(result["a_observed"])
            a_null.append(result["a_null"])
        c_counts.append(result["c_count"])
        if result["instrument_inside"]:
            instrument_cases_inside += 1
            c_observed.append(result["c_observed"])
            c_null.append(result["c_null"])

    a_obs = aggregate_rates(a_observed, radii) if a_observed else {r: np.nan for r in radii}
    a_nul = aggregate_rates(a_null, radii) if a_null else {r: np.nan for r in radii}
    c_obs = aggregate_rates(c_observed, radii) if c_observed else {r: np.nan for r in radii}
    c_nul = aggregate_rates(c_null, radii) if c_null else {r: np.nan for r in radii}

    lines = [
        "# Task 2 oracle signal measurement", "",
        f"Command root: `{args.root}`; seed: {args.seed}; requested cases: {len(cases)}; "
        f"loaded: {loaded}; fully skipped: {skipped}; en-face convention: `{args.enface_convention}`; "
        f"radii mode: `{args.radii_mode}`.", "",
        "The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by "
        "`src/fido/data/common.py`; all en-face measurements use that actual grid.", "",
        "## Geometry self-test", "",
        f"Crosshair maximum error: **{fmt(self_test_error if self_test_error is not None else np.nan)} px** "
        "(n=1; required <0.01 px).", "",
        "## A - Fundus vasculature landmarks versus OCT vessels", "",
        f"Cases with volume segmentation: n={len(a_counts)}; cases contributing >=1 in-footprint "
        f"landmark: n={len(a_observed)}. In-footprint landmarks per case: mean="
        f"{fmt(float(np.mean(a_counts)) if a_counts else np.nan)}, median="
        f"{fmt(float(np.median(a_counts)) if a_counts else np.nan)}.", "",
        f"| radius ({'px' if args.radii_mode == 'pixels' else 'fraction'}) | observed | uniform null | observed/null |", "|---:|---:|---:|---:|",
    ]
    for radius in radii:
        ratio = a_obs[radius] / a_nul[radius] if a_nul[radius] > 0 else np.nan
        lines.append(f"| {radius} | {fmt(a_obs[radius])} | {fmt(a_nul[radius])} | {fmt(ratio)} |")
    lines += ["", "## B - Intensity projection signal", "",
              f"Random-translation null uses {NULL_SAMPLES} positions per case. Values are case means; "
              f"n is reported per projection. Segmentation missing in {seg_missing} loaded case(s).", "",
              "| projection | n | GT NCC | NCC z-score | GT MI | null MI |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in b_values.items():
        array = np.asarray(values)
        means = np.array([finite_mean(array[:, column]) for column in range(array.shape[1])])
        valid_n = int(np.sum(np.all(np.isfinite(array), axis=1)))
        lines.append(f"| {name} | {valid_n} | {fmt(means[0])} | {fmt(means[1])} | {fmt(means[2])} | {fmt(means[3])} |")
    lines += ["", "## C - Instrument landmarks versus OCT instrument", "",
              "The OCT mask is the union of official classes 8 (`Forceps`), 10 "
              "(`Endoilluminator`), and 11 (`InstrumentInOCT`); class 12 "
              "(`ToolMirrorOCTArtifact`) is excluded. This measures shared instrument presence; "
              "it does not require assigning an ambiguous generic class to a particular tool.", "",
              f"Cases with segmentation: n={len(c_counts)}; cases with >=1 instrument keypoint inside "
              f"the OCT footprint: {instrument_cases_inside}/{len(c_counts)} "
              f"({fmt(instrument_cases_inside / len(c_counts) if c_counts else np.nan)}). "
              f"Cases contributing overlap rates: n={len(c_observed)}.", "",
              f"| radius ({'px' if args.radii_mode == 'pixels' else 'fraction'}) | observed | uniform null | observed/null |", "|---:|---:|---:|---:|"]
    for radius in radii:
        ratio = c_obs[radius] / c_nul[radius] if c_nul[radius] > 0 else np.nan
        lines.append(f"| {radius} | {fmt(c_obs[radius])} | {fmt(c_nul[radius])} | {fmt(ratio)} |")
    if warnings_log:
        lines += ["", "## Warnings", ""] + [f"- {message}" for message in warnings_log]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Task 2 oracle measurement: requested={len(cases)} loaded={loaded} skipped={skipped}")
    print(f"Geometry self-test: max_error={fmt(self_test_error if self_test_error is not None else np.nan)} px (n=1)")
    print(f"A: n_cases={len(a_counts)} n_with_points={len(a_observed)} mean_points={fmt(float(np.mean(a_counts)) if a_counts else np.nan)}")
    print(f"B: n_cases={loaded} projections={len(b_values)} null_positions={NULL_SAMPLES}/case")
    print(f"C: n_cases={len(c_counts)} cases_with_tip={instrument_cases_inside} fraction={fmt(instrument_cases_inside / len(c_counts) if c_counts else np.nan)}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
