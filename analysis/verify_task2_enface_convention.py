"""Verify the Task 2 uv-to-en-face convention under all eight square symmetries."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.data.common import load_volume_label_maps  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402

VESSEL_CLASS = 3
INSTRUMENT_CLASSES = (8, 10, 11)
RADII = (0.005, 0.01, 0.02, 0.04)
NULL_SAMPLES_PER_POINT = 200
VARIANTS = tuple(
    (f"{'transpose' if transpose else 'identity'}__"
     f"{'flip_u' if flip_u else 'keep_u'}__{'flip_v' if flip_v else 'keep_v'}",
     transpose, flip_u, flip_v)
    for transpose in (False, True)
    for flip_u in (False, True)
    for flip_v in (False, True)
)


def map_fundus_to_uv(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points))])
    uv_h = (np.linalg.inv(matrix) @ points_h.T).T
    return uv_h[:, :2] / uv_h[:, 2:3]


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


def transform_uv(uv: np.ndarray, transpose: bool, flip_u: bool, flip_v: bool) -> np.ndarray:
    transformed = uv.copy()
    if flip_u:
        transformed[:, 0] = 1.0 - transformed[:, 0]
    if flip_v:
        transformed[:, 1] = 1.0 - transformed[:, 1]
    return transformed[:, ::-1] if transpose else transformed


def sample_distances(uv: np.ndarray, distance: np.ndarray) -> np.ndarray:
    h, w = distance.shape
    pixels = uv * np.array([w - 1, h - 1], dtype=np.float64)
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, h - 1)
    return distance[y, x]


def unit_square_distance(mask: np.ndarray) -> np.ndarray:
    """Distance in fractions of the square side, not native-grid pixels."""
    h, w = mask.shape
    return distance_transform_edt(~mask.astype(bool), sampling=(1.0 / (h - 1), 1.0 / (w - 1)))


def measure_points(uv: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> dict[str, dict]:
    distance = unit_square_distance(mask)
    null_uv = rng.random((len(uv) * NULL_SAMPLES_PER_POINT, 2))
    measured = {}
    for name, transpose, flip_u, flip_v in VARIANTS:
        observed = sample_distances(transform_uv(uv, transpose, flip_u, flip_v), distance)
        null = sample_distances(transform_uv(null_uv, transpose, flip_u, flip_v), distance)
        measured[name] = {"observed": observed, "null": null}
    return measured


def measure_case(case: dict, case_index: int, seed: int) -> dict:
    case_id = f"{case['scenario']}/{case['frame_id']}"
    result = {"case_id": case_id, "loaded": False, "error": None}
    try:
        rng = np.random.default_rng(seed + case_index)
        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
        volume_dir = case["scenario_dir"] / "iOCT Microscope" / "Volume" / case["frame_id"]
        seg_dir = volume_dir / "Segmentation"
        if not seg_dir.is_dir() or not any(seg_dir.glob("*.png")):
            raise FileNotFoundError("volume Segmentation is missing")
        seg = load_volume_label_maps(seg_dir)
        result["loaded"] = True

        vessel_uv = inside_uv(points_from_group(data, "Vasculature"), matrix)
        instrument_points = np.concatenate([
            points_from_group(data, "Endgripping Forceps",
                              ("Right Head Tip", "Left Head Tip", "Joint Tip", "Start")),
            points_from_group(data, "Endoilluminator", ("Tip", "Start")),
        ])
        instrument_uv = inside_uv(instrument_points, matrix)
        result["a_count"] = len(vessel_uv)
        result["c_count"] = len(instrument_uv)
        result["a"] = (measure_points(vessel_uv, np.any(seg == VESSEL_CLASS, axis=1), rng)
                       if len(vessel_uv) else None)
        result["c"] = (measure_points(instrument_uv, np.any(np.isin(seg, INSTRUMENT_CLASSES), axis=1), rng)
                       if len(instrument_uv) else None)
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def aggregate(results: list[dict], key: str) -> dict[str, dict]:
    output = {}
    for name, *_ in VARIANTS:
        observed = [result[key][name]["observed"] for result in results if result.get(key)]
        null = [result[key][name]["null"] for result in results if result.get(key)]
        obs_values = np.concatenate(observed) if observed else np.empty(0)
        null_values = np.concatenate(null) if null else np.empty(0)
        output[name] = {
            "n": len(obs_values), "null_n": len(null_values),
            "rates": {
                radius: (float(np.mean(obs_values <= radius)) if len(obs_values) else np.nan,
                         float(np.mean(null_values <= radius)) if len(null_values) else np.nan)
                for radius in RADII
            },
        }
    return output


def ratio(observed: float, null: float) -> float:
    return observed / null if null > 0 else np.nan


def fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.4f}"


def classify(c_metrics: dict[str, dict]) -> tuple[str, str]:
    small = np.asarray([
        [ratio(*c_metrics[name]["rates"][radius]) for radius in RADII[:2]]
        for name, *_ in VARIANTS
    ])
    consistently_high = np.all(small >= 1.5, axis=1)
    if consistently_high.sum() == 1:
        best = int(np.flatnonzero(consistently_high)[0])
        name = VARIANTS[best][0]
        return "axis_convention", (f"Una sola variante ({name}) alcanza ratio observado/null >= 1.5 "
                                   "en ambos radios chicos; el arnés estaba usando la convención incorrecta.")
    if np.all(consistently_high):
        return "anisotropic_radius", ("Las ocho variantes suben en conjunto al expresar el radio en unidades físicas; "
                                      "el problema era el radio anisótropo, no el eje.")
    return "not_convention", ("Ninguna variante muestra enriquecimiento >= 1.5 consistente en los dos radios chicos; "
                              "la convención no explica el resultado y la refutación de T2-R11 se sostiene.")


def table(title: str, metrics: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", "",
             "Cada celda de radio es `observado / null / ratio`; las tasas son pooled por keypoint.", "",
             "| variante | n puntos | n null | " + " | ".join(f"r={r:g}" for r in RADII) + " |",
             "|---|---:|---:|" + "---:|" * len(RADII)]
    for name, *_ in VARIANTS:
        metric = metrics[name]
        cells = []
        for radius in RADII:
            observed, null = metric["rates"][radius]
            cells.append(f"{fmt(observed)} / {fmt(null)} / {fmt(ratio(observed, null))}")
        lines.append(f"| {name} | {metric['n']} | {metric['null_n']} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/data/Task2"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    cases = find_task2_cases(args.root)
    if args.limit is not None:
        cases = cases[:args.limit]

    started = time.monotonic()
    ordered: list[dict | None] = [None] * len(cases)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(measure_case, case, index, args.seed): index
                   for index, case in enumerate(cases)}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                case = cases[index]
                result = {"case_id": f"{case['scenario']}/{case['frame_id']}", "loaded": False,
                          "error": f"worker process failure: {exc}"}
            ordered[index] = result
            status = "ok" if result["error"] is None else f"skipped: {result['error']}"
            print(f"[{completed}/{len(cases)}] {result['case_id']} {status}", flush=True)

    valid = []
    for result in ordered:
        assert result is not None
        if result["error"]:
            warnings.warn(f"{result['case_id']}: {result['error']}")
        else:
            valid.append(result)
    a_metrics, c_metrics = aggregate(valid, "a"), aggregate(valid, "c")
    verdict, explanation = classify(c_metrics)
    instrument_cases = sum(result["c_count"] > 0 for result in valid)
    fraction = instrument_cases / len(valid) if valid else np.nan

    lines = ["# Verificación de la convención en-face de Task 2", "",
             f"Root: `{args.root}`. Casos solicitados: n={len(cases)}; casos válidos con segmentación: "
             f"n={len(valid)}; omitidos: n={len(cases) - len(valid)}. Seed={args.seed}; "
             f"null={NULL_SAMPLES_PER_POINT} muestras uniformes por keypoint.", "",
             "Los radios 0.005, 0.01, 0.02 y 0.04 son fracciones del lado del cuadrado unitario. "
             "La transformada de distancia usa el espaciado físico de cada eje de la rejilla 128 x 512.", "",
             "## Interpretación pre-registrada", "",
             "- Si una variante da ratio observado/null >= 1.5 consistentemente en radios chicos y las otras "
             "quedan en ~1.0, esa es la convención correcta y el arnés estaba roto.",
             "- Si las ocho quedan en ~1.0, la convención no es el problema y la refutación de T2-R11 es sólida.",
             "- Si todas suben parejo al corregir la anisotropía, el bug era el radio, no el eje.", "",
             "## C — Instrumento", "",
             f"Casos con >=1 keypoint de instrumento dentro de la huella: {instrument_cases}/{len(valid)} "
             f"({fmt(fraction)}; n={len(valid)}).", ""]
    lines += table("C — Tasas por variante", c_metrics)
    lines += [""] + table("A — Vasculatura", a_metrics)
    lines += ["", "## Veredicto", "", f"**{verdict}** — {explanation}", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.monotonic() - started
    print(f"Task 2 en-face verification: requested={len(cases)} valid={len(valid)} skipped={len(cases)-len(valid)} n={len(cases)}")
    print(f"C footprint: cases_with_keypoint={instrument_cases}/{len(valid)} fraction={fmt(fraction)} n={len(valid)}")
    for name, *_ in VARIANTS:
        values = " ".join(
            f"r={radius:g}:obs={fmt(c_metrics[name]['rates'][radius][0])},null={fmt(c_metrics[name]['rates'][radius][1])},ratio={fmt(ratio(*c_metrics[name]['rates'][radius]))}"
            for radius in RADII
        )
        print(f"C {name}: n={c_metrics[name]['n']} null_n={c_metrics[name]['null_n']} {values}")
    for name, *_ in VARIANTS:
        values = " ".join(
            f"r={radius:g}:ratio={fmt(ratio(*a_metrics[name]['rates'][radius]))}"
            for radius in RADII
        )
        print(f"A {name}: n={a_metrics[name]['n']} null_n={a_metrics[name]['null_n']} {values}")
    print(f"Verdict: {verdict} — {explanation}")
    print(f"Elapsed: {elapsed:.1f}s; report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
