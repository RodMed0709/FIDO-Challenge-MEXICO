"""Evaluacion all-case de Task 1 con la metrica oficial 0..10."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fido.geometry import MAX_THRESHOLD_PX, corner_auc

DEFAULT_DISTANCE_FALLBACK = 159.2


def _as_scenarios(scenarios, n: int) -> np.ndarray:
    values = np.asarray(scenarios, dtype=str)
    if values.shape != (n,):
        raise ValueError(f"scenarios must have shape ({n},), got {values.shape}")
    return values


def _summarize(errors: np.ndarray, scenarios: np.ndarray) -> dict:
    if errors.ndim != 1 or not np.isfinite(errors).all():
        raise ValueError("errors must be a finite one-dimensional array")
    cdf = np.asarray([np.mean(errors <= threshold)
                      for threshold in range(MAX_THRESHOLD_PX + 1)], dtype=np.float64)
    per_scenario = {}
    for scenario in np.unique(scenarios):
        selected = errors[scenarios == scenario]
        per_scenario[str(scenario)] = {
            "n": int(selected.size),
            "auc": float(corner_auc(selected)),
            "mean_error": float(selected.mean()),
            "cdf": np.asarray([np.mean(selected <= threshold)
                               for threshold in range(MAX_THRESHOLD_PX + 1)]),
        }
    return {
        "n": int(errors.size),
        "auc": float(corner_auc(errors)),
        "mean_error": float(errors.mean()),
        "cdf": cdf,
        "errors": errors,
        "per_scenario": per_scenario,
    }


def evaluate_keypoints(pred_xy, gt_xy, scenarios) -> dict:
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"pred_xy and gt_xy must have matching (N, 2) shape; got {pred.shape}, {gt.shape}")
    scenario_values = _as_scenarios(scenarios, len(pred))
    out = _summarize(np.linalg.norm(pred - gt, axis=1), scenario_values)
    out.update({"predictions": pred, "ground_truth": gt, "scenarios": scenario_values})
    return out


def evaluate_distances(pred, gt, scenarios, measured_mask,
                       fallback: float = DEFAULT_DISTANCE_FALLBACK) -> dict:
    predictions = np.asarray(pred, dtype=np.float64).copy()
    ground_truth = np.asarray(gt, dtype=np.float64)
    measured = np.asarray(measured_mask, dtype=bool)
    if predictions.shape != ground_truth.shape or predictions.ndim != 1:
        raise ValueError("pred and gt must have matching one-dimensional shapes")
    if measured.shape != predictions.shape:
        raise ValueError("measured_mask must match pred shape")
    scenarios_array = _as_scenarios(scenarios, len(predictions))
    predictions[~measured] = fallback
    if not np.isfinite(predictions).all() or not np.isfinite(ground_truth).all():
        raise ValueError("pred and gt must be finite after fallback")
    out = _summarize(np.abs(predictions - ground_truth), scenarios_array)
    out.update({
        "coverage": float(measured.mean()),
        "measured_mask": measured,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "scenarios": scenarios_array,
        "fallback": float(fallback),
    })
    for scenario, metrics in out["per_scenario"].items():
        scenario_mask = scenarios_array == scenario
        metrics["coverage"] = float(measured[scenario_mask].mean())
    return out


def write_case_predictions(path: Path, rows: list[dict]) -> None:
    """Escribe predicciones por caso en CSV o JSONL sin agregar denominadores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    elif path.suffix.lower() == ".csv":
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("prediction output must end in .csv or .jsonl")


def write_evaluation_outputs(path: Path, metrics: dict,
                             case_ids=None) -> tuple[Path, Path]:
    """Guarda una fila por caso y un JSON sidecar pooled/per-scenario."""
    predictions = np.asarray(metrics["predictions"])
    ground_truth = np.asarray(metrics["ground_truth"])
    errors = np.asarray(metrics["errors"])
    scenarios = np.asarray(metrics["scenarios"])
    ids = (np.asarray(case_ids, dtype=str) if case_ids is not None
           else np.asarray([f"{scenario}/{i}" for i, scenario in enumerate(scenarios)]))
    if ids.shape != (len(errors),):
        raise ValueError("case_ids must match evaluation length")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("case_ids must be unique")
    rows = []
    for i in range(len(errors)):
        row = {
            "case_id": str(ids[i]),
            "scenario": str(scenarios[i]),
            "prediction": json.dumps(np.asarray(predictions[i]).tolist()),
            "ground_truth": json.dumps(np.asarray(ground_truth[i]).tolist()),
            "error": float(errors[i]),
        }
        if "measured_mask" in metrics:
            row["measured"] = bool(metrics["measured_mask"][i])
        rows.append(row)
    write_case_predictions(path, rows)

    metrics_path = path.with_suffix(".metrics.json")
    summary = {
        "n": metrics["n"],
        "auc": metrics["auc"],
        "mean_error": metrics["mean_error"],
        "cdf": np.asarray(metrics["cdf"]).tolist(),
        "per_scenario": {
            scenario: {
                key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
                for key, value in values.items()
            }
            for scenario, values in metrics["per_scenario"].items()
        },
    }
    if "coverage" in metrics:
        summary["coverage"] = metrics["coverage"]
        summary["fallback"] = metrics["fallback"]
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, metrics_path
