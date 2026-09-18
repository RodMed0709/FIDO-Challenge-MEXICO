from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PositivePairs:
    positive: torch.Tensor
    fundus: torch.Tensor
    oct: torch.Tensor
    fundus_xy: torch.Tensor
    oct_xy: torch.Tensor
    batch_index: torch.Tensor


def _grid_xy(height: int, width: int, *, device, dtype) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def sample_positive_pairs(
    fundus_desc: torch.Tensor,
    oct_desc: torch.Tensor,
    gt_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int = 64,
    *,
    fundus_image_size: tuple[int, int] | None = None,
    generator: torch.Generator | None = None,
) -> PositivePairs:
    """Sample GT-aligned dense pairs.

    ``gt_matrix`` is ``M_native @ C`` and maps canonical OCT coordinates ``(u,v)`` in [0, 1] to
    fundus pixels. ``oct_desc`` must already use the dataset's canonical
    ``transpose__flip_u__flip_v`` convention, where width is u and height is v.
    Returned coordinates are in original fundus pixels and canonical OCT
    descriptor-grid pixels, respectively.
    """
    if fundus_desc.ndim != 4 or oct_desc.ndim != 4:
        raise ValueError("descriptor maps must be BCHW")
    b, channels, fh, fw = fundus_desc.shape
    if oct_desc.shape[:2] != (b, channels) or gt_matrix.shape != (b, 3, 3):
        raise ValueError("batch/channel or matrix shape mismatch")
    if valid_mask.shape != (b, 1, fh, fw):
        raise ValueError("valid_mask must match the fundus descriptor grid")
    image_h, image_w = fundus_image_size or (fh, fw)
    oh, ow = oct_desc.shape[-2:]
    oct_grid = _grid_xy(oh, ow, device=oct_desc.device, dtype=oct_desc.dtype)
    canonical = oct_grid.clone()
    canonical[:, 0] /= max(ow - 1, 1)
    canonical[:, 1] /= max(oh - 1, 1)
    homogeneous = torch.cat((canonical, torch.ones_like(canonical[:, :1])), dim=1)

    all_fundus, all_oct, all_fxy, all_oxy, all_batch = [], [], [], [], []
    for batch in range(b):
        mapped = (gt_matrix[batch].to(homogeneous) @ homogeneous.T).T[:, :2]
        feature_xy = mapped.clone()
        feature_xy[:, 0] *= (fw - 1) / max(image_w - 1, 1)
        feature_xy[:, 1] *= (fh - 1) / max(image_h - 1, 1)
        rounded = feature_xy.round().long()
        inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < fw) &
                  (rounded[:, 1] >= 0) & (rounded[:, 1] < fh))
        candidates = inside.nonzero(as_tuple=False).flatten()
        if candidates.numel():
            r = rounded[candidates]
            candidates = candidates[valid_mask[batch, 0, r[:, 1], r[:, 0]]]
        if candidates.numel() == 0:
            continue
        if candidates.numel() > k:
            order = torch.randperm(candidates.numel(), generator=generator)
            candidates = candidates[order[:k].to(candidates.device)]
        r = rounded[candidates]
        fundus_vectors = fundus_desc[batch, :, r[:, 1], r[:, 0]].T
        oct_vectors = oct_desc[batch].permute(1, 2, 0).reshape(-1, channels)[candidates]
        all_fundus.append(fundus_vectors)
        all_oct.append(oct_vectors)
        all_fxy.append(mapped[candidates])
        all_oxy.append(oct_grid[candidates])
        all_batch.append(torch.full((len(candidates),), batch, device=oct_desc.device, dtype=torch.long))
    if not all_fundus:
        empty = fundus_desc.new_empty((0, channels))
        empty_xy = fundus_desc.new_empty((0, 2))
        return PositivePairs(empty.new_empty((0,)), empty, empty, empty_xy, empty_xy,
                             torch.empty(0, device=fundus_desc.device, dtype=torch.long))
    fundus = torch.cat(all_fundus)
    oct_values = torch.cat(all_oct)
    return PositivePairs((fundus * oct_values).sum(1), fundus, oct_values,
                         torch.cat(all_fxy), torch.cat(all_oxy), torch.cat(all_batch))


def sample_intraimage_negatives(pairs: PositivePairs, fundus_desc: torch.Tensor,
                                valid_mask: torch.Tensor, count: int = 64,
                                exclusion_radius_px: float = 12.0,
                                fundus_image_size: tuple[int, int] | None = None) -> torch.Tensor:
    b, channels, fh, fw = fundus_desc.shape
    image_h, image_w = fundus_image_size or (fh, fw)
    grid = _grid_xy(fh, fw, device=fundus_desc.device, dtype=fundus_desc.dtype)
    grid[:, 0] *= (image_w - 1) / max(fw - 1, 1)
    grid[:, 1] *= (image_h - 1) / max(fh - 1, 1)
    negatives = []
    for index, batch in enumerate(pairs.batch_index.tolist()):
        allowed = valid_mask[batch, 0].flatten() & (
            torch.linalg.vector_norm(grid - pairs.fundus_xy[index], dim=1) >= exclusion_radius_px)
        indices = allowed.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("no valid intra-image negatives outside exclusion radius")
        indices = indices[torch.arange(count, device=indices.device) % indices.numel()]
        candidates = fundus_desc[batch].permute(1, 2, 0).reshape(-1, channels)[indices]
        negatives.append((candidates * pairs.oct[index]).sum(1))
    return torch.stack(negatives)


def dense_infonce(positive: torch.Tensor, negatives: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive = positive.reshape(-1, 1)
    if negatives.ndim == 1:
        negatives = negatives.unsqueeze(0).expand(len(positive), -1)
    if negatives.ndim != 2 or negatives.shape[0] != positive.shape[0]:
        raise ValueError("negatives must have shape (N,M) or (M,)")
    logits = torch.cat((positive, negatives), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), device=logits.device, dtype=torch.long))
