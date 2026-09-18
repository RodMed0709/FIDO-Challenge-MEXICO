#!/usr/bin/env python3
"""Diagnostica qué componentes explican el error del baseline Task 2."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fido.data.common import group_kfold_indices  # noqa: E402
from fido.data.task2 import Task2Dataset  # noqa: E402
from fido.eval_task2 import (  # noqa: E402
    evaluate_task2_cases, require_task2_training_root, write_case_jsonl,
)
from fido.geometry import compose_similarity  # noqa: E402
from fido.models.task2_baseline import FundusEnfaceHeatmapModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--enface-cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    root = require_task2_training_root(args.root)
    dataset = Task2Dataset(root, include_vessel_enface=False,
                           include_vessel_mask=False, enface_cache_dir=args.enface_cache)
    groups = [case["scenario"] for case in dataset.cases]
    splits = group_kfold_indices(groups, n_splits=args.n_folds, seed=0)
    for _ in range(args.fold + 1):
        _, val_idx = next(splits)
    loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    model = FundusEnfaceHeatmapModel(base_channels=32, n_downsamples=4)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    predicted = {key: [] for key in ("tx", "ty", "cos", "sin", "scale")}
    truth = {key: [] for key in ("tx", "ty", "cos", "sin", "scale")}
    gt_matrices, scenarios, case_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["fundus"].to(args.device), batch["enface"].to(args.device))
            target = batch["target_params"]
            for key, value in (("tx", output["tx"]), ("ty", output["ty"]),
                               ("cos", output["cos_theta"]), ("sin", output["sin_theta"]),
                               ("scale", output["scale"])):
                predicted[key].append(value.detach().cpu().numpy())
            for index, key in enumerate(("tx", "ty", "cos", "sin", "scale")):
                truth[key].append(target[:, index].numpy())
            gt_matrices.append(batch["gt_matrix"].numpy())
            scenarios.extend(batch["scenario"])
            case_ids.extend(f"{scenario}/{frame}" for scenario, frame in
                            zip(batch["scenario"], batch["frame_id"]))

    predicted = {key: np.concatenate(values) for key, values in predicted.items()}
    truth = {key: np.concatenate(values) for key, values in truth.items()}
    gt = np.concatenate(gt_matrices).astype(np.float64)
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("Duplicate case IDs detected; probable Task2Dataset I/O fallback")

    variants: dict[str, dict] = {}
    rows: list[dict] = []

    def evaluate(key: str, label: str, use_gt: tuple[str, ...] = ()) -> float:
        values = {name: (truth[name] if name in use_gt else predicted[name]).astype(np.float64)
                  for name in predicted}
        matrices = compose_similarity(values["tx"], values["ty"], values["cos"],
                                      values["sin"], values["scale"], reflect=True)
        result = evaluate_task2_cases(matrices, gt, scenarios, case_ids=case_ids)
        print(f"  {label:38s} error medio {result['mean_error']:7.2f} px   "
              f"mediana {result['median_error']:7.2f}   AUC {result['auc']:.4f}")
        variants[key] = {name: value for name, value in result.items()
                         if name not in {"errors", "cases"}}
        rows.extend({**row, "variant": key, "fold": args.fold} for row in result["cases"])
        return result["mean_error"]

    print(f"\nDiagnóstico sobre {len(gt)} casos de validación (fold {args.fold})\n")
    base = evaluate("baseline", "modelo tal cual (nada sustituido)")
    position = evaluate("gt_position", "con posición (tx,ty) perfecta", ("tx", "ty"))
    rotation = evaluate("gt_rotation", "con rotación (cos,sin) perfecta", ("cos", "sin"))
    scale = evaluate("gt_scale", "con escala perfecta", ("scale",))
    evaluate("gt_rotation_scale", "con rotación + escala perfectas", ("cos", "sin", "scale"))
    evaluate("gt_position_scale", "con posición + escala perfectas", ("tx", "ty", "scale"))
    evaluate("gt_position_rotation", "con posición + rotación perfectas", ("tx", "ty", "cos", "sin"))
    evaluate("all_gt", "TODO perfecto (control)", ("tx", "ty", "cos", "sin", "scale"))

    print("\nCaída del error al sustituir cada componente:")
    for label, value in (("posición", position), ("rotación", rotation), ("escala", scale)):
        fraction = (base - value) / base * 100 if base else 0.0
        print(f"  {label:20s} {base - value:7.2f} px ({fraction:5.1f}%)")

    angular = np.abs(np.arctan2(np.sin(np.arctan2(predicted["sin"], predicted["cos"]) -
                                          np.arctan2(truth["sin"], truth["cos"])),
                                 np.cos(np.arctan2(predicted["sin"], predicted["cos"]) -
                                        np.arctan2(truth["sin"], truth["cos"]))))
    relative_scale = np.abs(predicted["scale"] - truth["scale"]) / truth["scale"]
    position_error = np.hypot(predicted["tx"] - truth["tx"], predicted["ty"] - truth["ty"])
    print(f"Error angular: mediana {np.degrees(np.median(angular)):.1f}°")
    print(f"Error de escala: mediana {np.median(relative_scale) * 100:.1f}%")
    print(f"Error de posición: mediana {np.median(position_error):.1f} px")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"fold": args.fold, "n_cases": len(gt),
                                           "checkpoint": str(args.checkpoint),
                                           "variants": variants}, indent=2,
                                          ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if args.jsonl:
        write_case_jsonl(args.jsonl, rows)


if __name__ == "__main__":
    main()
