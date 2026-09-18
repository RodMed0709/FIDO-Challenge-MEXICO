"""Cache and compare Task 1 heatmap decoders on one frozen validation fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fido.data.task1 import Task1Dataset, task1_split
from fido.eval_task1 import evaluate_keypoints
from fido.heatmap_decode import decode_heatmap
from fido.models.task1_keypoint import Task1KeypointModel


def _state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint must be a state_dict or contain 'state_dict'")
    return {key.removeprefix("module."): value for key, value in loaded.items()}


def _build_cnn(state: dict[str, torch.Tensor]) -> Task1KeypointModel:
    first_key = "encoder.in_conv.block.0.weight"
    if first_key not in state:
        raise ValueError("checkpoint is not the clean Task1KeypointModel CNN")
    base_channels = int(state[first_key].shape[0])
    downsample_ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"encoder\.downsamples\.(\d+)\.", key))
    }
    model = Task1KeypointModel(base_channels=base_channels, n_downsamples=len(downsample_ids))
    model.load_state_dict(state, strict=True)
    return model


def cache_fold_logits(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dataset = Task1Dataset(args.root, load_fundus=True, load_bscan=False,
                           cache_path=args.case_cache)
    _, validation = task1_split(dataset.cases, n_splits=args.n_folds,
                                fold=args.fold, seed=args.seed)
    indices = list(validation.indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    model = _build_cnn(_state_dict(args.checkpoint)).to(args.device).eval()
    logits, ground_truth, scenarios, case_ids = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["fundus"].to(args.device))
            logits.append(output["heatmap_logits"].cpu().numpy())
            ground_truth.append(batch["keypoint"].numpy())
            scenarios.extend(str(value) for value in batch["scenario"])
            case_ids.extend(
                f"{scenario}/{frame}" for scenario, frame in
                zip(batch["scenario"], batch["frame_id"])
            )
    if not logits:
        raise RuntimeError("validation fold is empty")
    arrays = {
        "logits": np.concatenate(logits),
        "ground_truth": np.concatenate(ground_truth),
        "scenarios": np.asarray(scenarios),
        "case_ids": np.asarray(case_ids),
        "image_hw": np.asarray([1024, 1024], dtype=np.int64),
        "metadata_json": np.asarray(json.dumps({
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "fold": args.fold,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "validation_indices": indices,
            "coordinate_mapping": "legacy_stride",
        }, sort_keys=True)),
    }
    if len(set(case_ids)) != len(indices) or len(case_ids) != len(indices):
        raise RuntimeError("cache denominator/identity mismatch")
    args.cache_logits.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache_logits, **arrays)
    return arrays


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"logits", "ground_truth", "scenarios", "case_ids", "image_hw",
                    "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"logit cache is missing: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
    n = len(arrays["logits"])
    if any(len(arrays[key]) != n for key in ("ground_truth", "scenarios", "case_ids")):
        raise ValueError("logit cache arrays have inconsistent denominators")
    if len(set(arrays["case_ids"].tolist())) != n:
        raise ValueError("case_ids in logit cache must be unique")
    return arrays


def summarize_decoder(arrays: dict[str, np.ndarray], mode: str,
                      temperature: float, window: int) -> dict:
    logits = torch.from_numpy(arrays["logits"])
    heat_h, heat_w = logits.shape[-2:]
    image_h, image_w = arrays["image_hw"].tolist()
    decoded = decode_heatmap(logits, mode=mode, temperature=temperature, window=window)
    scale = decoded.new_tensor([image_w / heat_w, image_h / heat_h])
    predictions = (decoded[:, 0] * scale).numpy()
    metrics = evaluate_keypoints(predictions, arrays["ground_truth"], arrays["scenarios"])
    errors = metrics["errors"]
    return {
        "mode": mode,
        "n": metrics["n"],
        "auc": metrics["auc"],
        "mean_error": metrics["mean_error"],
        "cdf": metrics["cdf"].tolist(),
        "pck": {str(t): float(np.mean(errors <= t)) for t in (1, 3, 5, 10)},
        "per_scenario": {
            scenario: {
                "n": values["n"], "auc": values["auc"],
                "mean_error": values["mean_error"], "cdf": values["cdf"].tolist(),
            }
            for scenario, values in metrics["per_scenario"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--logits", type=Path)
    parser.add_argument("--cache-logits", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/root/data_cache/Task1"))
    parser.add_argument("--case-cache", type=Path,
                        default=Path("/root/data_cache/task1_cases_local.json"))
    parser.add_argument("--mode", choices=("global", "local", "dark"), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        if args.cache_logits is None:
            parser.error("--checkpoint requires --cache-logits")
        arrays = cache_fold_logits(args)
    else:
        arrays = load_cache(args.logits)
    selected = summarize_decoder(arrays, args.mode, args.temperature, args.window)
    global_metrics = (selected if args.mode == "global" else
                      summarize_decoder(arrays, "global", args.temperature, args.window))
    result = {
        "cache": str(args.cache_logits or args.logits),
        "cache_metadata": json.loads(str(arrays["metadata_json"])),
        "selected": selected,
        "global_control": global_metrics,
        "delta_auc_vs_global": selected["auc"] - global_metrics["auc"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
