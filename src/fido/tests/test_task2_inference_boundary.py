import numpy as np
import pytest
import torch

from fido.data.common import TASK2_CANONICAL_TO_NATIVE, canonicalize_task2_matrix
from fido.inference_task2 import (
    CanonicalTask2Bundle,
    export_canonical_task2_matrix,
    inference,
)


class _Predictor:
    def __init__(self, canonical):
        self.canonical = canonical

    def predict_canonical(self, oct_volume, opmi_image):
        assert oct_volume.shape == (2, 3, 4)
        assert opmi_image.shape == (8, 8, 3)
        return self.canonical


def test_nativeize_round_trip_recovers_official_native_matrix():
    native = np.array([[3.0, 4.0, 100.0], [4.0, -3.0, 200.0], [0.0, 0.0, 1.0]])
    canonical = canonicalize_task2_matrix(native)
    exported = export_canonical_task2_matrix(torch.from_numpy(canonical))
    assert exported.dtype == np.float64
    assert np.array_equal(exported, native)
    assert np.array_equal(canonical, native @ TASK2_CANONICAL_TO_NATIVE)


def test_inference_returns_official_native_float64_matrix():
    native = np.array([[3.0, 4.0, 100.0], [4.0, -3.0, 200.0], [0.0, 0.0, 1.0]])
    bundle = CanonicalTask2Bundle(_Predictor(canonicalize_task2_matrix(native)), np.eye(3))
    result = inference(1, np.zeros((2, 3, 4), np.uint8),
                       np.zeros((8, 8, 3), np.uint8), bundle)
    assert result.dtype == np.float64
    assert result.shape == (3, 3)
    assert np.array_equal(result, native)


def test_inference_none_uses_native_fallback_without_conversion():
    fallback = np.array([[1.0, 0.0, 12.0], [0.0, -1.0, 34.0], [0.0, 0.0, 1.0]])
    bundle = CanonicalTask2Bundle(_Predictor(np.eye(3)), fallback)
    assert np.array_equal(inference(1, None, np.zeros((8, 8, 3)), bundle), fallback)


def test_boundary_rejects_wrong_task_and_non_homogeneous_output():
    bundle = CanonicalTask2Bundle(_Predictor(np.eye(3)), np.eye(3))
    with pytest.raises(ValueError, match="task_id"):
        inference(0, None, np.zeros((8, 8, 3)), bundle)
    with pytest.raises(ValueError, match="homogeneous"):
        export_canonical_task2_matrix(np.diag([1.0, 1.0, 2.0]))
