"""Evaluación reproducible de Task 2 contra la geometría oficial.

Este módulo no carga datos ni modelos. Recibe matrices ya predichas para que
diagnósticos distintos compartan exactamente la misma agregación pooled, por
escenario y por caso.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from fido.geometry import compose_similarity, corner_auc, corner_error, decompose_similarity

_COMPONENTS = frozenset({"position", "rotation", "scale"})


def require_task2_training_root(root: str | Path) -> Path:
    """Prohíbe usar etiquetas Mock en cualquier diagnóstico por caso."""
    resolved = Path(root)
    normalized_parts = {part.lower().replace("_", " ").replace("-", " ")
                        for part in resolved.parts}
    if "mock test" in normalized_parts:
        raise RuntimeError("Mock Test labels are forbidden for per-case Task 2 diagnostics")
    return resolved


def _matrices(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (N, 3, 3); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def evaluate_task2_cases(
    pred_matrix,
    gt_matrix,
    scenarios: Sequence[str],
    *,
    case_ids: Sequence[str] | None = None,
) -> dict:
    """Calcula error/AUC pooled, por escenario y filas auditables por caso."""
    pred = _matrices(pred_matrix, "pred_matrix")
    gt = _matrices(gt_matrix, "gt_matrix")
    scenario_values = np.asarray(scenarios, dtype=str).reshape(-1)
    n_cases = len(pred)
    if len(gt) != n_cases or len(scenario_values) != n_cases:
        raise ValueError("pred_matrix, gt_matrix, and scenarios must have the same number of cases")

    if case_ids is None:
        ids = [str(index) for index in range(n_cases)]
    else:
        ids = [str(value) for value in case_ids]
        if len(ids) != n_cases:
            raise ValueError("case_ids must have the same number of cases")
        if len(set(ids)) != len(ids):
            raise ValueError("case_ids must be unique; duplicate dataset samples detected")

    errors = np.asarray(corner_error(pred, gt), dtype=np.float64)
    cases = [
        {"case_id": ids[index], "scenario": str(scenario_values[index]),
         "corner_error": float(errors[index])}
        for index in range(n_cases)
    ]
    per_scenario = {}
    for scenario in dict.fromkeys(scenario_values.tolist()):
        scenario_errors = errors[scenario_values == scenario]
        per_scenario[str(scenario)] = {
            "n_cases": int(len(scenario_errors)),
            "mean_error": float(np.mean(scenario_errors)),
            "median_error": float(np.median(scenario_errors)),
            "auc": float(corner_auc(scenario_errors)),
        }
    return {
        "n_cases": n_cases,
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "auc": float(corner_auc(errors)),
        "errors": errors,
        "per_scenario": per_scenario,
        "cases": cases,
    }


def substitute_gt_components(pred_matrix, gt_matrix, components: Iterable[str]) -> np.ndarray:
    """Sustituye posición, rotación y/o escala GT en predicciones 4-DOF."""
    pred = _matrices(pred_matrix, "pred_matrix")
    gt = _matrices(gt_matrix, "gt_matrix")
    if len(pred) != len(gt):
        raise ValueError("pred_matrix and gt_matrix must have the same number of cases")
    selected = frozenset(components)
    unknown = selected - _COMPONENTS
    if unknown:
        raise ValueError(f"unknown components: {sorted(unknown)}")
    if selected == _COMPONENTS:
        return gt.copy()

    pred_params = decompose_similarity(pred)
    gt_params = decompose_similarity(gt)
    source = dict(pred_params)
    if "position" in selected:
        source["tx"], source["ty"] = gt_params["tx"], gt_params["ty"]
    if "rotation" in selected:
        source["cos_theta"] = gt_params["cos_theta"]
        source["sin_theta"] = gt_params["sin_theta"]
    if "scale" in selected:
        source["scale"] = gt_params["scale"]
    return np.asarray(compose_similarity(**source, reflect=True), dtype=np.float64)


def write_case_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Escribe filas por caso de forma determinista y JSON estándar."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    keys = [(row.get("variant", row.get("condition", "default")), row.get("case_id"))
            for row in materialized]
    if any(case_id is None for _, case_id in keys) or len(set(keys)) != len(keys):
        raise ValueError("JSONL rows must have one unique case_id per variant/condition")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
