#!/usr/bin/env python3
"""T1-81: ceiling and checkpoint baseline for distance, fold-clean/all-case.

This script only reads Task 1 training data.  It deliberately does not know a
Mock Test path.  Calibration and fallback come from the checkpoint's
train-only ``distance_calibration.json``; validation is GroupKFold fold 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def ground_truth_distance(annotation: dict) -> float:
    """Official Task 1 JSON stores distance multiplied by ten."""
    return float(annotation["Ground Truth"]["Task 1"][2]) / 10.0


def validate_calibration_provenance(fit_case_ids, train_case_ids, val_case_ids) -> None:
    fit_ids = tuple(str(value) for value in fit_case_ids)
    train_ids = tuple(str(value) for value in train_case_ids)
    val_ids = set(str(value) for value in val_case_ids)
    if len(set(fit_ids)) != len(fit_ids):
        raise RuntimeError("Calibration fit_case_ids contain duplicates")
    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("Train split case IDs contain duplicates")
    fit_set, train_set = set(fit_ids), set(train_ids)
    missing, extra = train_set - fit_set, fit_set - train_set
    leaked = fit_set & val_ids
    if missing or extra or leaked:
        raise RuntimeError(
            "Calibration provenance does not exactly match train split: "
            f"missing={len(missing)}, extra={len(extra)}, val_overlap={len(leaked)}"
        )


def checkpoint_metadata(raw) -> dict:
    if not isinstance(raw, dict) or "state_dict" not in raw:
        return {}
    allowed = ("fold", "n_folds", "seed", "epoch", "best_epoch", "commit", "git_commit")
    return {key: raw[key] for key in allowed
            if key in raw and isinstance(raw[key], (str, int, float, bool, type(None)))}


def _load_case(root: Path, case: dict):
    frame_dir = root / case["scenario"] / "iOCT Microscope" / "Bscan" / case["frame_id"]
    images, masks = [], []
    if not frame_dir.is_dir():
        return torch.zeros(2, 512, 512), [None, None], False
    for stem in ("00", "01"):
        image_path = frame_dir / f"{stem}.png"
        mask_path = frame_dir / "Segmentation" / f"{stem}.png"
        image = (np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
                 if image_path.exists() else np.zeros((512, 512), np.float32))
        mask = np.asarray(Image.open(mask_path)) if mask_path.exists() else None
        images.append(image)
        masks.append(mask)
    return torch.from_numpy(np.stack(images)), masks, True


def _oracle_gap(mask, instrument_class=11, ilm_class=1):
    if mask is None:
        return float("nan")
    ys, xs = np.nonzero(mask == instrument_class)
    if not len(ys):
        return float("nan")
    deepest = int(np.argmax(ys))
    tip_row, tip_col = int(ys[deepest]), int(xs[deepest])
    low, high = max(0, tip_col - 5), min(mask.shape[1], tip_col + 6)
    ilm_rows = np.nonzero(mask[:, low:high] == ilm_class)[0]
    return float(ilm_rows.min() - tip_row) if len(ilm_rows) else float("nan")


def _markdown_metrics(title: str, metrics: dict) -> str:
    lines = [f"### {title}", "", "| subset | n | coverage | AUC | mean error |",
             "|---|---:|---:|---:|---:|"]
    lines.append(f"| pooled | {metrics['n']:,} | {metrics['coverage']:.4f} | "
                 f"{metrics['auc']:.6f} | {metrics['mean_error']:.3f} |")
    for scenario, row in sorted(metrics["per_scenario"].items()):
        lines.append(f"| {scenario} | {row['n']:,} | {row['coverage']:.4f} | "
                     f"{row['auc']:.6f} | {row['mean_error']:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/data_cache/Task1"))
    parser.add_argument("--case-cache", type=Path,
                        default=Path("/root/data_cache/task1_cases_local.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        help="Default: distance_calibration.json beside checkpoint")
    parser.add_argument("--repo", type=Path, default=Path("/workspace/FIDO_CHALLENGE"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "src"))
    from fido.data.task1 import find_task1_cases_cached, task1_split
    from fido.eval_task1 import evaluate_distances
    from fido.models.unet_bscan_seg import UNet, distance_from_segmentation
    from fido.train.train_task1_unet import load_distance_calibration

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    calibration_path = args.calibration or args.checkpoint.with_name("distance_calibration.json")
    for path, label in ((args.root, "root"), (args.case_cache, "case cache"),
                        (args.checkpoint, "checkpoint"), (calibration_path, "calibration")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    cases = find_task1_cases_cached(args.root, cache_path=args.case_cache)
    train_split, val_split = task1_split(
        cases, n_splits=args.n_folds, fold=args.fold, seed=args.seed
    )
    train_idx, val_idx = train_split.indices, val_split.indices
    train_ids = tuple(f"{cases[int(i)]['scenario']}/{cases[int(i)]['frame_id']}" for i in train_idx)
    val_ids = tuple(f"{cases[int(i)]['scenario']}/{cases[int(i)]['frame_id']}" for i in val_idx)
    calibration = load_distance_calibration(calibration_path)
    validate_calibration_provenance(calibration.fit_case_ids, train_ids, val_ids)

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_metadata = checkpoint_metadata(raw)
    state = raw.get("state_dict", raw)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    base = state["encoders.0.conv1.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("encoders."))
    model = UNet(base_channels=base, depth=depth)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device).eval()

    gt, scenarios, oracle_pred, oracle_measured = [], [], [], []
    model_pred, model_measured, pending_images, pending_positions = [], [], [], []
    start = time.time()

    def flush():
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(device)
        n = len(batch)
        with torch.inference_mode():
            logits = model(batch.reshape(n * 2, 1, 512, 512)).reshape(n, 2, 3, 512, 512)
        for j, position in enumerate(pending_positions):
            gaps = [distance_from_segmentation(logits[j:j + 1, s]).item() for s in range(2)]
            gap = mean_finite(gaps)
            if np.isfinite(gap):
                model_pred[position] = calibration.scale * gap + calibration.offset
                model_measured[position] = True
        pending_images.clear(); pending_positions.clear()

    for count, index in enumerate(val_idx, 1):
        case = cases[int(index)]
        annotation = json.loads((args.root / case["scenario"] / "Numerical" /
                                 f"{case['frame_id']}.json").read_text(encoding="utf-8"))
        gt.append(ground_truth_distance(annotation))
        scenarios.append(case["scenario"])
        images, masks, has_oct = _load_case(args.root, case)
        gap = mean_finite([_oracle_gap(mask) for mask in masks])
        oracle_pred.append(calibration.scale * gap + calibration.offset if np.isfinite(gap)
                           else calibration.fallback)
        oracle_measured.append(bool(np.isfinite(gap)))
        model_pred.append(calibration.fallback)
        model_measured.append(False)
        if has_oct:
            pending_images.append(images); pending_positions.append(count - 1)
        if len(pending_images) >= args.batch_size:
            flush()
        if count % 1000 == 0:
            print(f"[validation] {count:,}/{len(val_idx):,} ({time.time()-start:.1f}s)", flush=True)
    flush()

    oracle = evaluate_distances(oracle_pred, gt, scenarios, oracle_measured,
                                fallback=calibration.fallback)
    baseline = evaluate_distances(model_pred, gt, scenarios, model_measured,
                                  fallback=calibration.fallback)
    train_scenarios = sorted({cases[int(i)]["scenario"] for i in train_idx})
    val_scenarios = sorted(set(scenarios))
    report = f"""# T1-81 — techo real y baseline honesto de distancia

**Estado:** ejecutado sobre train Task 1; Mock Test no usado.  
**Semilla:** `{args.seed}`; **split:** GroupKFold `{args.n_folds}`, fold `{args.fold}`.  
**Train scenarios:** `{', '.join(train_scenarios)}`.  
**Validation scenarios:** `{', '.join(val_scenarios)}`.  
**Checkpoint:** `{args.checkpoint}` (`sha256:{checkpoint_sha256(args.checkpoint)}`).  
**Calibration:** `{calibration_path}` (`sha256:{checkpoint_sha256(calibration_path)}`).  
**Calibration train-only:** scale `{calibration.scale:.8f}`, offset `{calibration.offset:.8f}`, fallback `{calibration.fallback:.8f}`, `{len(calibration.fit_case_ids):,}` fit IDs (cobertura exacta del train fold).  
**Metadata embebida del checkpoint:** `{json.dumps(saved_metadata, sort_keys=True)}`.  

La contradicción `308 vs 376.7 px` era conceptual: `376.7 = 0.732·511+2.6779`
es el máximo algebraico de una separación axial de 511 filas, no evidencia de
que una punta/ILM sea visible y medible. El techo real se mide con máscaras GT,
manteniendo en el denominador cada caso no medible y aplicando el fallback
ajustado exclusivamente en train.

{_markdown_metrics('Oráculo geométrico con máscaras GT', oracle)}

{_markdown_metrics('Checkpoint real', baseline)}

## Gate

`oracle_distance_auc_all = {oracle['auc']:.6f}`.  
`distance_auc_all = {baseline['auc']:.6f}`.  
`coverage_oracle = {oracle['coverage']:.6f}`; `coverage_checkpoint = {baseline['coverage']:.6f}`.  

Decisión: **{'conservar geometría como candidato' if oracle['auc'] >= .35 else 'priorizar cabeza aprendida' if oracle['auc'] < .25 else 'zona intermedia; no promover sin comparación T1-89'}**.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
