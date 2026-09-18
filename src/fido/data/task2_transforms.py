from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from fido.geometry import decompose_similarity


def _size_hw(size: tuple[int, int]) -> tuple[int, int]:
    h, w = (int(size[0]), int(size[1]))
    if h < 2 or w < 2:
        raise ValueError("image dimensions must be at least 2")
    return h, w


def pixel_crop_resize_homography(
    crop_xywh: tuple[float, float, float, float], output_size: tuple[int, int]
) -> torch.Tensor:
    """Map source pixel coordinates in a crop to output pixel coordinates."""
    x, y, crop_w, crop_h = map(float, crop_xywh)
    out_h, out_w = _size_hw(output_size)
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("crop width and height must be positive")
    return torch.tensor(
        [[out_w / crop_w, 0.0, -x * out_w / crop_w],
         [0.0, out_h / crop_h, -y * out_h / crop_h],
         [0.0, 0.0, 1.0]], dtype=torch.float32,
    )


def pixel_rotation_homography(
    angle_deg: float, image_size: tuple[int, int], scale: float = 1.0
) -> torch.Tensor:
    """Proper rotation/scale about the pixel-center of an image (old -> new)."""
    h, w = _size_hw(image_size)
    if scale <= 0:
        raise ValueError("scale must be positive")
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    linear = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    center = torch.tensor([cx, cy], dtype=torch.float32)
    translation = center - linear @ center
    result = torch.eye(3, dtype=torch.float32)
    result[:2, :2] = linear
    result[:2, 2] = translation
    return result


def _pixel_to_normalized(height: int, width: int, *, dtype, device) -> torch.Tensor:
    return torch.tensor(
        [[2.0 / (width - 1), 0.0, -1.0],
         [0.0, 2.0 / (height - 1), -1.0],
         [0.0, 0.0, 1.0]], dtype=dtype, device=device,
    )


def apply_fundus_homography(
    fundus: torch.Tensor,
    gt_matrix: torch.Tensor,
    image_h: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp CHW fundus and compose its pixel-space GT as ``image_h @ M``.

    The returned boolean mask identifies output pixels sampled from real input
    support, so padded rotation/crop pixels cannot become correspondences.
    """
    if fundus.ndim != 3:
        raise ValueError("fundus must be CHW")
    in_h, in_w = fundus.shape[-2:]
    out_h, out_w = output_size or (in_h, in_w)
    out_h, out_w = _size_hw((out_h, out_w))
    hmat = torch.as_tensor(image_h, dtype=fundus.dtype, device=fundus.device)
    if hmat.shape != (3, 3) or not torch.isfinite(hmat).all():
        raise ValueError("image_h must be a finite 3x3 matrix")
    if not torch.allclose(hmat[2], hmat.new_tensor([0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("only affine pixel homographies are supported")
    n_in = _pixel_to_normalized(in_h, in_w, dtype=fundus.dtype, device=fundus.device)
    n_out = _pixel_to_normalized(out_h, out_w, dtype=fundus.dtype, device=fundus.device)
    output_to_input = n_in @ torch.linalg.inv(hmat) @ torch.linalg.inv(n_out)
    theta = output_to_input[:2].unsqueeze(0)
    grid = F.affine_grid(theta, (1, fundus.shape[0], out_h, out_w), align_corners=True)
    warped = F.grid_sample(fundus.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True).squeeze(0)
    # Test the continuous sampling coordinate, not a nearest-warped ones mask:
    # nearest interpolation would incorrectly admit a half-pixel halo whose
    # bilinear image value already mixes real data and zero padding.
    valid = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)).squeeze(0).unsqueeze(0)
    matrix = hmat.to(dtype=gt_matrix.dtype, device=gt_matrix.device) @ gt_matrix
    return warped, matrix, valid


@dataclass(frozen=True)
class ScaleAugmentationConfig:
    factor_min: float
    factor_max: float
    train_log_scale_q05: float
    train_log_scale_q95: float
    source_indices: tuple[int, ...]


def derive_scale_augmentation(scales, train_idx, extrapolation: float = 0.25) -> ScaleAugmentationConfig:
    """Derive multiplicative scale support exclusively from explicit train rows."""
    values = np.asarray(scales, dtype=np.float64)
    indices = np.asarray(train_idx, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("train_idx must contain at least two rows")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("train_idx must contain unique in-range indices")
    selected = values[indices]
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise ValueError("training scales must be finite and positive")
    q05, q95 = np.quantile(np.log(selected), [0.05, 0.95])
    half_span = max(float(q95 - q05) / 2.0, 1e-6) * (1.0 + float(extrapolation))
    return ScaleAugmentationConfig(
        factor_min=float(math.exp(-half_span)), factor_max=float(math.exp(half_span)),
        train_log_scale_q05=float(q05), train_log_scale_q95=float(q95),
        source_indices=tuple(int(i) for i in indices),
    )


class Task2GeometricAugment:
    """Deterministic per-(seed,index) scale augmentation of the fundus only."""

    def __init__(self, factor_min: float, factor_max: float, seed: int = 0):
        if not 0 < factor_min <= 1 <= factor_max:
            raise ValueError("scale range must bracket 1")
        self.factor_min, self.factor_max, self.seed = factor_min, factor_max, int(seed)

    def __call__(self, sample: Mapping, sample_index: int) -> dict:
        generator = torch.Generator().manual_seed(self.seed + int(sample_index))
        u = torch.rand((), generator=generator).item()
        factor = math.exp(math.log(self.factor_min) * (1 - u) + math.log(self.factor_max) * u)
        fundus = sample["fundus"]
        hmat = pixel_rotation_homography(0.0, fundus.shape[-2:], scale=factor)
        warped, matrix, valid = apply_fundus_homography(fundus, sample["gt_matrix"], hmat)
        params = decompose_similarity(matrix, reflect=False)
        out = dict(sample)
        out.update(fundus=warped, gt_matrix=matrix, valid_mask=valid,
                   fundus_homography=hmat.to(matrix),
                   target_params=torch.stack([params[k] for k in ("tx", "ty", "cos_theta", "sin_theta", "scale")]))
        return out
