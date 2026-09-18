import numpy as np
import torch

from fido.data.task2_transforms import (
    Task2GeometricAugment,
    apply_fundus_homography,
    derive_scale_augmentation,
    pixel_crop_resize_homography,
    pixel_rotation_homography,
)
from fido.geometry import compose_similarity, corner_error, project_corners
from analysis.evaluate_task2_scale_holdout import build_scale_holdout, summarize_predictions
from fido.models.task2_baseline import FundusEnfaceHeatmapModel


def _reflected_matrix():
    return torch.as_tensor(
        compose_similarity(300.0, 420.0, np.cos(0.3), np.sin(0.3), 160.0, reflect=True),
        dtype=torch.float32,
    )


def test_fundus_transform_composes_ground_truth_and_corners():
    image = torch.zeros(3, 32, 32)
    matrix = _reflected_matrix()
    h = pixel_rotation_homography(12.0, image_size=(32, 32))
    _, matrix2, valid = apply_fundus_homography(image, matrix, h)

    expected = h @ matrix
    assert torch.allclose(matrix2, expected, atol=1e-5)
    assert corner_error(matrix2, expected).item() < 1e-3
    assert torch.allclose(project_corners(matrix2), project_corners(h @ matrix), atol=1e-4)
    assert torch.det(matrix2[:2, :2]) < 0
    assert valid.dtype == torch.bool and valid.shape == (1, 32, 32)


def test_crop_resize_maps_crop_corners_exactly():
    h = pixel_crop_resize_homography((10, 20, 50, 40), output_size=(80, 100))
    points = torch.tensor([[10.0, 20.0, 1.0], [60.0, 60.0, 1.0]])
    mapped = (h @ points.T).T[:, :2]
    assert torch.allclose(mapped, torch.tensor([[0.0, 0.0], [100.0, 80.0]]))


def test_composed_crop_resize_rotate_updates_matrix_exactly():
    image = torch.rand(3, 64, 64)
    matrix = _reflected_matrix()
    crop = pixel_crop_resize_homography((8, 6, 48, 52), (64, 64))
    rotate = pixel_rotation_homography(-17.0, (64, 64))
    h = rotate @ crop
    _, matrix2, valid = apply_fundus_homography(image, matrix, h)
    assert torch.allclose(matrix2, h @ matrix, atol=1e-5)
    assert 0 < valid.sum() < valid.numel()


def test_scale_range_uses_only_train_indices():
    scales = np.array([100.0, 110.0, 120.0, 10_000.0])
    config = derive_scale_augmentation(scales, train_idx=np.array([0, 1, 2]))
    changed_val = scales.copy()
    changed_val[3] = 1_000_000.0
    config2 = derive_scale_augmentation(changed_val, train_idx=np.array([0, 1, 2]))
    assert config == config2
    assert config.source_indices == (0, 1, 2)
    assert 0 < config.factor_min < 1 < config.factor_max


def test_deterministic_augment_preserves_valid_mask_and_target_contract():
    sample = {"fundus": torch.ones(3, 32, 32), "gt_matrix": _reflected_matrix()}
    augment = Task2GeometricAugment(0.8, 1.2, seed=7)
    out = augment(sample, sample_index=13)
    assert out["valid_mask"].dtype == torch.bool
    assert torch.allclose(out["gt_matrix"], out["fundus_homography"] @ sample["gt_matrix"])
    assert torch.det(out["gt_matrix"][:2, :2]) < 0


def test_holdout_contract_is_unchanged_by_extreme_validation_scale():
    scales = np.array([130.0, 145.0, 160.0, 175.0, 190.0, 250.0])
    train_idx, val_idx = np.arange(5), np.array([5])
    first = build_scale_holdout(scales, train_idx, val_idx)
    scales[-1] = 1e12
    second = build_scale_holdout(scales, train_idx, val_idx)
    assert first["augmentation"] == second["augmentation"]
    assert first["synthetic_scale_targets_px"] == second["synthetic_scale_targets_px"]


def test_invalid_pixels_cannot_change_model_output_or_correlation():
    torch.manual_seed(4)
    model = FundusEnfaceHeatmapModel(base_channels=2, n_downsamples=2).eval()
    fundus = torch.rand(1, 3, 64, 64)
    enface = torch.rand(1, 1, 32, 32)
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    valid[:, :, :16] = False
    changed = fundus.clone()
    changed[:, :, :16] = torch.rand_like(changed[:, :, :16]) * 1000
    with torch.no_grad():
        first = model(fundus, enface, fundus_size=64, valid_mask=valid)
        second = model(changed, enface, fundus_size=64, valid_mask=valid)
    for key in ("heatmap_logits", "tx", "ty", "scale", "cos_theta", "sin_theta"):
        assert torch.equal(first[key], second[key])


def test_holdout_summary_reports_scale_and_per_scenario_gate_metrics():
    gt = np.stack([compose_similarity(10, 20, 1, 0, 100, True),
                   compose_similarity(30, 40, 1, 0, 200, True)])
    pred = np.stack([compose_similarity(10, 20, 1, 0, 90, True),
                     compose_similarity(30, 40, 1, 0, 220, True)])
    result = summarize_predictions(pred, gt, np.array(["S1", "S2"]))
    assert result["scale_mae_px"] == 15.0
    assert result["scale_relative_mae"] == 0.1
    assert result["per_scenario"]["S1"]["scale_mae_px"] == 10.0
    assert "auc" in result["per_scenario"]["S2"]
