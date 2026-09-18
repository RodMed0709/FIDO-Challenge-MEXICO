"""Build the leak-free synthetic scale holdout specification for T2-81.

This script deliberately uses training annotations only. Model evaluation is
performed after the baseline and scale-augmented checkpoints exist; until then
the emitted JSON is the frozen holdout contract, not a claimed result.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fido.data.common import group_kfold_indices  # noqa: E402
from fido.data.task2 import Task2Dataset, load_task2_scales  # noqa: E402
from fido.data.task2_transforms import derive_scale_augmentation  # noqa: E402
from fido.data.task2_transforms import apply_fundus_homography, pixel_rotation_homography  # noqa: E402
from fido.eval_task2 import evaluate_task2_cases  # noqa: E402
from fido.geometry import compose_similarity, decompose_similarity  # noqa: E402
from fido.models.task2_baseline import FundusEnfaceHeatmapModel  # noqa: E402


def build_scale_holdout(scales, train_idx, val_idx, extrapolation: float = 0.25) -> dict:
    config = derive_scale_augmentation(scales, train_idx, extrapolation)
    train = np.asarray(scales, dtype=np.float64)[np.asarray(train_idx)]
    center = float(np.exp(np.median(np.log(train))))
    # The two bands are outside the robust train support and fixed before any
    # checkpoint prediction is inspected.
    low = center * config.factor_min
    high = center * config.factor_max
    return {
        "augmentation": asdict(config),
        "synthetic_scale_targets_px": [low, high],
        "synthetic_scale_factors": [config.factor_min, config.factor_max],
        "train_indices": [int(i) for i in train_idx],
        "validation_indices": [int(i) for i in val_idx],
        "selection_uses_mock": False,
        "gate": {
            "scale_error_relative_reduction_min": 0.20,
            "corner_auc_must_increase": True,
            "max_per_scenario_auc_drop": 0.01,
        },
    }


def summarize_predictions(predictions, targets, scenarios) -> dict:
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(targets, dtype=np.float64)
    result = evaluate_task2_cases(pred, gt, scenarios)
    pred_scale = np.asarray(decompose_similarity(pred)["scale"])
    gt_scale = np.asarray(decompose_similarity(gt)["scale"])
    scale_error = np.abs(pred_scale - gt_scale)
    relative = scale_error / gt_scale
    result.update(scale_mae_px=float(scale_error.mean()), scale_median_px=float(np.median(scale_error)),
                  scale_relative_mae=float(relative.mean()))
    for scenario in result["per_scenario"]:
        mask = np.asarray(scenarios) == scenario
        result["per_scenario"][scenario]["scale_mae_px"] = float(scale_error[mask].mean())
        result["per_scenario"][scenario]["scale_relative_mae"] = float(relative[mask].mean())
    return {key: value for key, value in result.items() if key not in {"errors", "cases"}}


class _FixedScaleSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, factor: float):
        self.dataset, self.indices, self.factor = dataset, tuple(indices), float(factor)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        sample = self.dataset[self.indices[item]]
        hmat = pixel_rotation_homography(0, sample["fundus"].shape[-2:], self.factor)
        fundus, matrix, valid = apply_fundus_homography(sample["fundus"], sample["gt_matrix"], hmat)
        sample.update(fundus=fundus, gt_matrix=matrix, valid_mask=valid)
        return sample


def evaluate_checkpoint(checkpoint: Path, dataset, val_idx, factors, device, batch_size,
                        num_workers) -> dict:
    model = FundusEnfaceHeatmapModel(base_channels=32, n_downsamples=4)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    all_pred, all_gt, all_scenarios = [], [], []
    with torch.inference_mode():
        for factor in factors:
            loader = DataLoader(_FixedScaleSubset(dataset, val_idx, factor), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
            for batch in loader:
                fundus, enface = batch["fundus"].to(device), batch["enface"].to(device)
                output = model(fundus, enface, valid_mask=batch["valid_mask"].to(device))
                pred = compose_similarity(output["tx"], output["ty"], output["cos_theta"],
                                          output["sin_theta"], output["scale"], reflect=True)
                all_pred.extend(pred.cpu().numpy())
                all_gt.extend(batch["gt_matrix"].numpy())
                all_scenarios.extend(batch["scenario"])
    return summarize_predictions(all_pred, all_gt, all_scenarios)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-extrapolation", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--augmented-checkpoint", type=Path)
    parser.add_argument("--enface-cache", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if "mock test" in str(args.root).lower():
        raise RuntimeError("Mock Test labels are forbidden for T2-81")
    dataset = Task2Dataset(args.root, include_vessel_enface=False, include_vessel_mask=False,
                           enface_cache_dir=args.enface_cache)
    splits = group_kfold_indices([c["scenario"] for c in dataset.cases], args.n_folds, args.seed)
    for _ in range(args.fold + 1):
        train_idx, val_idx = next(splits)
    result = build_scale_holdout(load_task2_scales(dataset.cases), train_idx, val_idx,
                                 args.scale_extrapolation)
    supplied = (args.baseline_checkpoint is not None, args.augmented_checkpoint is not None)
    if supplied[0] != supplied[1]:
        parser.error("baseline-checkpoint and augmented-checkpoint must be supplied together")
    if all(supplied):
        factors = result["synthetic_scale_factors"]
        baseline = evaluate_checkpoint(args.baseline_checkpoint, dataset, val_idx, factors,
                                       args.device, args.batch_size, args.num_workers)
        augmented = evaluate_checkpoint(args.augmented_checkpoint, dataset, val_idx, factors,
                                        args.device, args.batch_size, args.num_workers)
        scenario_drops = [baseline["per_scenario"][key]["auc"] -
                          augmented["per_scenario"][key]["auc"] for key in baseline["per_scenario"]]
        reduction = 1.0 - augmented["scale_mae_px"] / max(baseline["scale_mae_px"], 1e-12)
        result.update(baseline=baseline, augmented=augmented,
                      observed_scale_error_relative_reduction=float(reduction),
                      observed_max_per_scenario_auc_drop=float(max(scenario_drops, default=0.0)),
                      gate_pass=bool(augmented["auc"] > baseline["auc"] and reduction >= 0.20 and
                                     max(scenario_drops, default=0.0) <= 0.01))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
