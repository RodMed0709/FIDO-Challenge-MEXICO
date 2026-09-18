"""Versioned Task 2 boundary for future canonical-coordinate pipelines.

This module deliberately contains no solver. A trained pipeline supplies a
``predict_canonical`` method; this boundary performs the mandatory conversion
to the official native OCT frame immediately before returning from inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from fido.data.common import nativeize_task2_matrix

TASK2_PIPELINE_SCHEMA = "task2_canonical_v1"


class CanonicalTask2Predictor(Protocol):
    def predict_canonical(self, oct_volume: np.ndarray, opmi_image: np.ndarray): ...


@dataclass
class CanonicalTask2Bundle:
    predictor: CanonicalTask2Predictor
    fallback_native: np.ndarray
    schema: str = TASK2_PIPELINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TASK2_PIPELINE_SCHEMA:
            raise ValueError(f"unsupported Task 2 pipeline schema: {self.schema}")
        self.fallback_native = _official_matrix(self.fallback_native)


def _numpy_matrix(matrix) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    array = np.asarray(matrix)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("Task 2 prediction must be a finite (3,3) matrix")
    return array


def _official_matrix(matrix) -> np.ndarray:
    array = _numpy_matrix(matrix).astype(np.float64, copy=True)
    if not np.allclose(array[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Task 2 prediction must have homogeneous row [0,0,1]")
    array[2] = [0.0, 0.0, 1.0]
    return array


def export_canonical_task2_matrix(canonical_matrix) -> np.ndarray:
    """The only supported canonical→official export boundary."""
    canonical = _numpy_matrix(canonical_matrix)
    return _official_matrix(nativeize_task2_matrix(canonical))


def inference(task_id, oct_volume, opmi_image, model: CanonicalTask2Bundle):
    """Challenge-compatible entry point for a future trained canonical solver."""
    if int(task_id) != 1:
        raise ValueError(f"canonical Task 2 bundle cannot serve task_id={task_id}")
    if not isinstance(model, CanonicalTask2Bundle):
        raise TypeError("model must be a CanonicalTask2Bundle")
    if oct_volume is None:
        return model.fallback_native.copy()
    canonical = model.predictor.predict_canonical(oct_volume, opmi_image)
    return export_canonical_task2_matrix(canonical)
