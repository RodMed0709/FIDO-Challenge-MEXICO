import json

import numpy as np
import torch

from fido.data.common import (
    TASK2_CANONICAL_TO_NATIVE,
    TASK2_ENFACE_CONVENTION,
    canonicalize_task2_enface,
    canonicalize_task2_matrix,
)
from fido.data.task2 import Task2Dataset
from fido.data.task2_transforms import apply_fundus_homography
from fido.losses.task2_contrastive import sample_positive_pairs
from fido.geometry import compose_similarity


def test_canonicalization_is_exact_transpose_then_double_flip():
    native = np.arange(3 * 5).reshape(3, 5)
    canonical = canonicalize_task2_enface(native)
    assert TASK2_ENFACE_CONVENTION == "transpose__flip_u__flip_v"
    assert canonical.shape == (5, 3)
    assert np.array_equal(canonical, native[::-1, ::-1].T)
    assert canonical[0, 0] == native[-1, -1]
    assert canonical[-1, -1] == native[0, 0]


def test_numpy_and_tensor_conventions_are_identical():
    native = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    expected = canonicalize_task2_enface(native)
    actual = canonicalize_task2_enface(torch.from_numpy(native)).numpy()
    assert np.array_equal(actual, expected)


def test_canonical_corner_matrix_matches_point_transform():
    # native (x,y) = (1-v, 1-u) for canonical (u,v).
    canonical_to_native = np.array(
        [[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    corners = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    mapped = (canonical_to_native @ corners.T).T[:, :2]
    assert np.array_equal(mapped, [[1, 1], [1, 0], [0, 0], [0, 1]])
    assert np.allclose(canonical_to_native @ canonical_to_native, np.eye(3))


def test_native_and_canonical_matrices_project_identical_points_and_corners():
    native_matrix = np.array([[30.0, 4.0, 100.0], [-3.0, 31.0, 200.0], [0, 0, 1]])
    canonical_matrix = canonicalize_task2_matrix(native_matrix)
    canonical_points = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                                 [.2, .7, 1.0]])
    native_points = (TASK2_CANONICAL_TO_NATIVE @ canonical_points.T).T
    assert np.allclose((canonical_matrix @ canonical_points.T).T,
                       (native_matrix @ native_points.T).T)


def test_fundus_augmentation_composes_after_canonicalization():
    native = torch.tensor([[30.0, 0.0, 100.0], [0.0, 30.0, 200.0], [0.0, 0.0, 1.0]])
    canonical = canonicalize_task2_matrix(native)
    image_h = torch.tensor([[1.1, 0.0, -5.0], [0.0, 1.1, 7.0], [0.0, 0.0, 1.0]])
    _, augmented, _ = apply_fundus_homography(torch.zeros(3, 16, 16), canonical, image_h)
    assert torch.allclose(augmented, image_h @ native @ native.new_tensor(TASK2_CANONICAL_TO_NATIVE))


def test_dataset_canonicalizes_legacy_native_cache_once(tmp_path):
    cache = tmp_path / "cache" / "Scenario_01"
    cache.mkdir(parents=True)
    native = np.arange(15, dtype=np.uint8).reshape(3, 5)
    np.savez_compressed(cache / "00000.npz", enface=native,
                        fundus=np.zeros((8, 8, 3), dtype=np.uint8))
    numerical = tmp_path / "Scenario_01" / "Numerical"
    numerical.mkdir(parents=True)
    annotation = numerical / "00000.json"
    native_matrix = np.array([[2.0, 0.0, 1.0], [0.0, -2.0, 6.0], [0.0, 0.0, 1.0]])
    annotation.write_text(json.dumps({"Ground Truth": {"Task 2": native_matrix.tolist()}}))
    dataset = Task2Dataset(tmp_path, include_vessel_enface=False, include_vessel_mask=False,
                           enface_cache_dir=tmp_path / "cache")
    dataset.cases = [{"scenario": "Scenario_01", "frame_id": "00000",
                      "scenario_dir": tmp_path / "Scenario_01", "json_path": annotation}]
    sample = dataset[0]
    assert sample["enface_convention"] == TASK2_ENFACE_CONVENTION
    assert sample["enface"].shape == (1, 5, 3)
    assert np.array_equal((sample["enface"][0].numpy() * 255).round().astype(np.uint8),
                          native[::-1, ::-1].T)
    assert torch.allclose(sample["native_gt_matrix"], torch.from_numpy(native_matrix).float())
    assert torch.allclose(sample["gt_matrix"],
                          torch.from_numpy(native_matrix @ TASK2_CANONICAL_TO_NATIVE).float())
    params = sample["target_params"]
    rebuilt = compose_similarity(*params, reflect=False)
    assert torch.allclose(rebuilt, sample["gt_matrix"])
    assert torch.det(sample["native_gt_matrix"][:2, :2]) < 0
    assert torch.det(sample["gt_matrix"][:2, :2]) > 0

    # Every canonical descriptor point is supervised at M_native @ C @ (u,v).
    fundus = torch.randn(1, 2, 9, 9)
    oct_desc = torch.randn(1, 2, 5, 3)
    pairs = sample_positive_pairs(fundus, oct_desc, sample["gt_matrix"].unsqueeze(0),
                                  torch.ones(1, 1, 9, 9, dtype=torch.bool), k=15)
    canonical = pairs.oct_xy / torch.tensor([2.0, 4.0])
    homogeneous = torch.cat((canonical, torch.ones(len(canonical), 1)), dim=1)
    expected = (sample["gt_matrix"] @ homogeneous.T).T[:, :2]
    assert torch.allclose(pairs.fundus_xy, expected)
