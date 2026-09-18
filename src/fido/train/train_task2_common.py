from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fido.data.common import group_kfold_indices
from fido.data.task2 import Task2Dataset, Task2TransformSubset, load_task2_scales
from fido.data.task2_transforms import Task2GeometricAugment, derive_scale_augmentation
from fido.losses.task2_contrastive import dense_infonce, sample_intraimage_negatives
from fido.losses.task2_contrastive import sample_positive_pairs
from fido.models.task2_common import Task2CommonModel
from fido.models.task2_dinov2 import DualDinoCommonEncoder
from fido.train.task2_dinov2_optim import build_dino_optimizer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_group_fold(groups, n_folds: int, fold: int, seed: int = 0):
    splits = list(group_kfold_indices(list(groups), n_folds, seed))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0,{len(splits) - 1}]")
    train_idx, val_idx = splits[fold]
    if set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[val_idx]):
        raise RuntimeError("GroupKFold leaked a scenario across train and validation")
    return train_idx, val_idx


def common_checkpoint_payload(model, *, epoch: int, seed: int, fold: int,
                              stats: dict, scale_config, encoder: str, phase: str,
                              optimizer_provenance: dict) -> dict:
    return {"model": model.state_dict(), "epoch": epoch, "seed": seed, "fold": fold,
            "train_stats": stats, "coordinate_frame": "task2_canonical_v1",
            "encoder": encoder, "phase": phase,
            "optimizer_provenance": optimizer_provenance,
            "scale_augmentation": asdict(scale_config)}


class ValidMicrobatchAccumulator:
    """Average gradients over valid microbatches, including short remainders."""
    def __init__(self, optimizer, parameters, target_count: int):
        if target_count < 1:
            raise ValueError("target_count must be >=1")
        self.optimizer = optimizer
        self.parameters = list(parameters)
        self.target_count = target_count
        self.valid_count = 0
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss):
        loss.backward()  # sum now; divide every gradient by exact window count at flush.
        self.valid_count += 1

    def ready(self) -> bool:
        return self.valid_count == self.target_count

    def step(self, scheduler=None) -> bool:
        if self.valid_count == 0:
            return False
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(self.valid_count)
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.valid_count = 0
        return True


def capture_rng_state(generator: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler_generator": generator.get_state()}


def restore_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["sampler_generator"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def attach_resume_state(checkpoint: dict, *, global_step: int, phase_epoch: int,
                        optimizer, scheduler, generator, best_phase_score: float,
                        phase_metrics: dict) -> dict:
    checkpoint.update(global_step=global_step, phase_epoch=phase_epoch,
                      optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                      rng_state=capture_rng_state(generator),
                      best_phase_score=float(best_phase_score), phase_metrics=phase_metrics)
    return checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_ids_sha256(cases: list[dict], indices) -> str:
    values = [f"{cases[int(index)]['scenario']}/{cases[int(index)]['frame_id']}"
              for index in indices]
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def build_training_recipe(args, model, train_loader_length: int, scale_config) -> dict:
    fundus_size = getattr(model, "fundus_input_size", None)
    oct_size = getattr(model, "oct_input_size", None)
    steps_per_epoch = int(np.ceil(train_loader_length / args.grad_accum))
    return {
        "schema": "task2_common_recipe_v1", "encoder": args.encoder,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "amp_dtype": args.amp_dtype, "fundus_input_size": fundus_size,
        "oct_input_size": oct_size, "epochs": args.epochs,
        "lp_epochs": args.lp_epochs, "ft_epochs": args.ft_epochs,
        "lr": args.lr, "head_lr": args.head_lr,
        "last_block_lr": args.last_block_lr, "layer_decay": args.layer_decay,
        "k": args.k, "temperature": args.temperature,
        "negative_count": 64, "negative_exclusion_radius_px": 12.0,
        "scheduler": "warmup_5pct_cosine_v1",
        "optimizer_steps_per_epoch": steps_per_epoch,
        "phase_total_steps": {"lp": args.lp_epochs * steps_per_epoch,
                              "partial_ft": args.ft_epochs * steps_per_epoch,
                              "cnn": args.epochs * steps_per_epoch},
        "scale_augmentation": asdict(scale_config),
    }


def validate_resume_payload(payload: dict, recipe: dict, *, encoder: str,
                            fold: int, seed: int) -> None:
    expected_values = {"coordinate_frame": "task2_canonical_v1", "encoder": encoder,
                       "fold": fold, "seed": seed,
                       "gate_schema": "task2_common_representation_v1"}
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise ValueError(f"resume checkpoint {key} mismatch")
    if payload.get("recipe") != recipe:
        raise ValueError("resume checkpoint recipe mismatch; flags cannot diverge")


def validate_dino_prerequisites(gate_path: Path, control_path: Path, control_checkpoint: Path,
                                *, fold: int, seed: int, validation_indices,
                                train_ids_sha256: str, validation_ids_sha256: str) -> tuple[dict, dict]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    required_gate = {"gate_pass", "selection_uses_mock", "train_case_ids_sha256",
                     "validation_case_ids_sha256", "expected_convention",
                     "canonical_to_native_matrix", "gate_schema"}
    if not required_gate <= gate.keys() or gate["gate_pass"] is not True or gate["selection_uses_mock"]:
        raise ValueError("T2-83 gate artifact is incomplete, failed, or used Mock")
    if gate["expected_convention"] != "transpose__flip_u__flip_v":
        raise ValueError("T2-83 convention does not match the canonical pipeline")
    authoritative_c = [[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    if gate["canonical_to_native_matrix"] != authoritative_c:
        raise ValueError("T2-83 canonical-to-native matrix is not authoritative C")
    if gate["gate_schema"] != "task2_enface_convention_gate_v1":
        raise ValueError("T2-83 gate schema mismatch")
    if gate.get("fold") != fold or gate.get("seed") != seed:
        raise ValueError("T2-83 fold/seed mismatch")
    if gate.get("validation_indices") != [int(index) for index in validation_indices]:
        raise ValueError("T2-83 validation IDs do not match this exact fold")
    if gate["train_case_ids_sha256"] != train_ids_sha256 or gate["validation_case_ids_sha256"] != validation_ids_sha256:
        raise ValueError("T2-83 train/validation case ID hashes mismatch current dataset")
    required_control = {"gate_pass", "selection_uses_mock", "fold", "seed",
                        "validation_indices", "coordinate_frame", "checkpoint_sha256",
                        "metrics", "gate_schema"}
    if not required_control <= control.keys() or control["gate_pass"] is not True or control["selection_uses_mock"]:
        raise ValueError("T2-82 control artifact is incomplete, failed, or used Mock")
    if control["fold"] != fold or control["seed"] != seed:
        raise ValueError("T2-82 fold/seed mismatch")
    if control["validation_indices"] != [int(index) for index in validation_indices]:
        raise ValueError("T2-82 validation IDs do not match this exact fold")
    if control["coordinate_frame"] != "task2_canonical_v1":
        raise ValueError("T2-82 coordinate frame mismatch")
    if len(control["checkpoint_sha256"]) != 64:
        raise ValueError("T2-82 checkpoint hash is invalid")
    if _sha256(control_checkpoint) != control["checkpoint_sha256"]:
        raise ValueError("T2-82 checkpoint bytes do not match checkpoint_sha256")
    if control["gate_schema"] != "task2_common_representation_v1":
        raise ValueError("T2-82 gate schema mismatch")
    if control.get("train_case_ids_sha256") != train_ids_sha256 or control.get("validation_case_ids_sha256") != validation_ids_sha256:
        raise ValueError("T2-82 case ID hashes mismatch current dataset")
    checkpoint = torch.load(control_checkpoint, map_location="cpu", weights_only=False)
    expected_checkpoint = {"coordinate_frame": "task2_canonical_v1",
                           "gate_schema": "task2_common_representation_v1",
                           "encoder": "cnn", "fold": fold, "seed": seed,
                           "validation_indices": [int(index) for index in validation_indices],
                           "train_case_ids_sha256": train_ids_sha256,
                           "validation_case_ids_sha256": validation_ids_sha256}
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"T2-82 checkpoint provenance mismatch: {key}")
    if not isinstance(checkpoint.get("recipe"), dict) or checkpoint["recipe"].get("schema") != "task2_common_recipe_v1":
        raise ValueError("T2-82 checkpoint recipe is missing or unversioned")
    gate["artifact_sha256"] = _sha256(gate_path)
    control["artifact_sha256"] = _sha256(control_path)
    return gate, control


def compute_train_stats(dataset: Task2Dataset, train_indices) -> dict:
    """Streaming pixel statistics over train rows only (never validation/Mock)."""
    fundus_sum = torch.zeros(3, dtype=torch.float64)
    fundus_sq = torch.zeros(3, dtype=torch.float64)
    fundus_n = 0
    oct_sum = oct_sq = torch.tensor(0.0, dtype=torch.float64)
    oct_n = 0
    for index in train_indices:
        sample = dataset[int(index)]
        fundus = sample["fundus"].double().reshape(3, -1)
        enface = sample["enface"].double().reshape(-1)
        fundus_sum += fundus.sum(1)
        fundus_sq += fundus.square().sum(1)
        fundus_n += fundus.shape[1]
        oct_sum += enface.sum()
        oct_sq += enface.square().sum()
        oct_n += enface.numel()
    fundus_mean = fundus_sum / fundus_n
    oct_mean = oct_sum / oct_n
    return {
        "source_indices": [int(i) for i in train_indices],
        "fundus_mean": fundus_mean.tolist(),
        "fundus_std": torch.sqrt((fundus_sq / fundus_n - fundus_mean.square()).clamp_min(1e-12)).tolist(),
        "oct_mean": [float(oct_mean)],
        "oct_std": [float(torch.sqrt((oct_sq / oct_n - oct_mean.square()).clamp_min(1e-12)))],
    }


class _NormalizeAugment:
    def __init__(self, stats: dict, augment=None):
        self.augment = augment
        self.fm = torch.tensor(stats["fundus_mean"]).view(3, 1, 1)
        self.fs = torch.tensor(stats["fundus_std"]).view(3, 1, 1)
        self.om = torch.tensor(stats["oct_mean"]).view(1, 1, 1)
        self.os = torch.tensor(stats["oct_std"]).view(1, 1, 1)

    def __call__(self, sample: dict, index: int) -> dict:
        out = self.augment(sample, index) if self.augment is not None else dict(sample)
        out["fundus"] = (out["fundus"] - self.fm) / self.fs
        out["enface"] = (out["enface"] - self.om) / self.os
        if "valid_mask" not in out:
            out["valid_mask"] = torch.ones_like(out["fundus"][:1], dtype=torch.bool)
        return out


class _DerangedValidation(torch.utils.data.Dataset):
    """Attach a global OCT derangement; invariant to DataLoader batch size."""
    def __init__(self, dataset: Task2TransformSubset):
        if len(dataset) < 2:
            raise ValueError("OCT derangement requires at least two validation cases")
        self.dataset = dataset
        self.source_positions = np.roll(np.arange(len(dataset)), 1)
        if np.any(self.source_positions == np.arange(len(dataset))):
            raise RuntimeError("OCT derangement contains an auto-pair")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        target = dict(self.dataset[index])
        source = self.dataset[int(self.source_positions[index])]
        target["enface_shuffled"] = source["enface"]
        target["shuffle_source_id"] = f"{source['scenario']}/{source['frame_id']}"
        target["case_id"] = f"{target['scenario']}/{target['frame_id']}"
        if target["case_id"] == target["shuffle_source_id"]:
            raise RuntimeError("OCT derangement produced an auto-pair")
        return target


def representation_metrics(model, loader, device, k: int) -> dict:
    model.eval()
    pooled = {"paired_ranks": [], "shuffled_ranks": [], "center_errors": [],
              "shuffle_center_errors": []}
    scenario_values: dict[str, dict[str, list[float]]] = {}
    forward_seconds, forward_cases = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            fundus = batch["fundus"].to(device)
            enface = batch["enface"].to(device)
            matrices = batch["gt_matrix"].to(device)
            valid = batch["valid_mask"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(fundus, enface, valid)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - started
            forward_cases += len(fundus)
            shuffled_output = model(fundus, batch["enface_shuffled"].to(device), valid)
            batch_records = []
            for oct_map, rank_name in ((output["oct_desc"], "paired_ranks"),
                                       (shuffled_output["oct_desc"], "shuffled_ranks")):
                pairs = sample_positive_pairs(output["fundus_desc"], oct_map, matrices,
                                              output["descriptor_valid_mask"], k=k,
                                              fundus_image_size=fundus.shape[-2:])
                local = [[] for _ in range(len(fundus))]
                for i in range(len(pairs.positive)):
                    candidates = output["fundus_desc"][pairs.batch_index[i]].flatten(1).T
                    scores = candidates @ pairs.oct[i]
                    local[int(pairs.batch_index[i])].append(
                        float((scores > pairs.positive[i]).sum().item() + 1))
                batch_records.append((rank_name, local))
            # M2: known rotation/scale; match the OCT center descriptor to fundus.
            for i in range(len(fundus)):
                oh, ow = output["oct_desc"].shape[-2:]
                candidates = output["fundus_desc"][i].flatten(1).T
                valid_flat = output["descriptor_valid_mask"][i, 0].flatten()
                fh, fw = output["fundus_desc"].shape[-2:]
                center = matrices[i, :2, :2] @ matrices.new_tensor([0.5, 0.5]) + matrices[i, :2, 2]
                record = scenario_values.setdefault(batch["scenario"][i],
                    {name: [] for name in pooled})
                for rank_name, local in batch_records:
                    pooled[rank_name].extend(local[i])
                    record[rank_name].extend(local[i])
                for desc, error_name in ((output["oct_desc"], "center_errors"),
                                         (shuffled_output["oct_desc"], "shuffle_center_errors")):
                    query = desc[i, :, oh // 2, ow // 2]
                    scores = (candidates @ query).masked_fill(~valid_flat, -torch.inf)
                    best = int(scores.argmax())
                    pred = torch.tensor([best % fw, best // fw], device=device) * torch.tensor(
                        [(fundus.shape[-1] - 1) / max(fw - 1, 1),
                         (fundus.shape[-2] - 1) / max(fh - 1, 1)], device=device)
                    error = float(torch.linalg.vector_norm(pred - center))
                    pooled[error_name].append(error)
                    record[error_name].append(error)
    candidate_count = output["fundus_desc"].shape[-2] * output["fundus_desc"].shape[-1]
    threshold = max(1, int(np.ceil(candidate_count * 0.01)))
    def summarize(values):
        paired = float(np.mean(np.asarray(values["paired_ranks"]) <= threshold))
        shuffled = float(np.mean(np.asarray(values["shuffled_ranks"]) <= threshold))
        errors = np.asarray(values["center_errors"])
        shuffle_errors = np.asarray(values["shuffle_center_errors"])
        return {"m1_top1pct": paired, "m1_oct_shuffle_top1pct": shuffled,
                "shuffle_drop_points": 100.0 * (paired - shuffled),
                "m2_center_10px": float(np.mean(errors <= 10)),
                "m2_median_px": float(np.median(errors)),
                "m2_oct_shuffle_10px": float(np.mean(shuffle_errors <= 10)),
                "m2_shuffle_drop_points": 100.0 * (np.mean(errors <= 10) - np.mean(shuffle_errors <= 10))}
    summary = summarize(pooled)
    summary["encoder_seconds_per_case"] = forward_seconds / max(forward_cases, 1)
    summary["per_scenario"] = {scenario: summarize(values)
                               for scenario, values in scenario_values.items()}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--enface-cache", type=Path, default=None)
    parser.add_argument("--encoder", choices=("cnn", "dinov2"), default="cnn")
    parser.add_argument("--stage", choices=("lpft",), default="lpft")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lp-epochs", type=int, default=3)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--last-block-lr", type=float, default=1e-5)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/task2_common_cnn"))
    parser.add_argument("--control-results", type=Path,
                        help="Frozen T2-82/83 results.json; required for the DINO relative gate.")
    parser.add_argument("--t2-83-gate", type=Path,
                        help="Frozen train-only T2-83 oracle JSON; required for DINO.")
    parser.add_argument("--control-checkpoint", type=Path,
                        help="T2-82 checkpoint whose bytes must match control-results.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-valid-steps", type=int, default=0,
                        help="Exit after N valid optimizer steps, reporting peak CUDA memory.")
    args = parser.parse_args()
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >=1")
    if args.encoder == "dinov2" and any(value is None for value in
                                         (args.control_results, args.control_checkpoint,
                                          args.t2_83_gate)):
        parser.error("--control-results, --control-checkpoint and --t2-83-gate are required for DINO")
    _seed_everything(args.seed)
    dataset = Task2Dataset(args.root, include_vessel_enface=False, include_vessel_mask=False,
                           enface_cache_dir=args.enface_cache)
    groups = [case["scenario"] for case in dataset.cases]
    train_idx, val_idx = select_group_fold(groups, args.n_folds, args.fold, args.seed)
    train_ids_hash = case_ids_sha256(dataset.cases, train_idx)
    validation_ids_hash = case_ids_sha256(dataset.cases, val_idx)
    prerequisite_gate = None
    if args.encoder == "dinov2":
        prerequisite_gate, _ = validate_dino_prerequisites(
            args.t2_83_gate, args.control_results, args.control_checkpoint,
            fold=args.fold, seed=args.seed,
            validation_indices=val_idx, train_ids_sha256=train_ids_hash,
            validation_ids_sha256=validation_ids_hash)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = compute_train_stats(dataset, train_idx)
    scales = load_task2_scales(dataset.cases)
    scale_config = derive_scale_augmentation(scales, train_idx)
    (args.out / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    augment = Task2GeometricAugment(scale_config.factor_min, scale_config.factor_max, args.seed)
    train_set = Task2TransformSubset(dataset, train_idx, _NormalizeAugment(stats, augment))
    val_set = Task2TransformSubset(dataset, val_idx, _NormalizeAugment(stats))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=generator)
    val_loader = DataLoader(_DerangedValidation(val_set), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    device = torch.device(args.device)
    model = (Task2CommonModel(128) if args.encoder == "cnn"
             else DualDinoCommonEncoder(128, pretrained=True)).to(device)
    recipe = build_training_recipe(args, model, len(train_loader), scale_config)
    log_path = args.out / "train.jsonl"
    phases = (("cnn", args.epochs),) if args.encoder == "cnn" else (
        ("lp", args.lp_epochs), ("partial_ft", args.ft_epochs))
    epoch = 0
    global_step = 0
    phase_metrics = {}
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_payload.get("resumable", True) is not True:
            raise ValueError("smoke checkpoints are intentionally not resumable mid-epoch")
        validate_resume_payload(resume_payload, recipe, encoder=args.encoder,
                                fold=args.fold, seed=args.seed)
        model.load_state_dict(resume_payload["model"])
        epoch = int(resume_payload["epoch"])
        global_step = int(resume_payload["global_step"])
        restore_rng_state(resume_payload["rng_state"], generator)
    for phase, phase_epochs in phases:
      optimizer = (torch.optim.AdamW(model.parameters(), lr=args.lr)
                   if phase == "cnn" else build_dino_optimizer(
                       model, phase, head_lr=args.head_lr, last_block_lr=args.last_block_lr,
                       layer_decay=args.layer_decay))
      optimizer_steps_per_epoch = max(1, int(np.ceil(len(train_loader) / args.grad_accum)))
      total_steps = max(1, phase_epochs * optimizer_steps_per_epoch)
      warmup_steps = max(1, int(total_steps * 0.05))
      scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer, lambda step: (step + 1) / warmup_steps if step < warmup_steps else
          0.5 * (1.0 + np.cos(np.pi * (step - warmup_steps) /
                                     max(total_steps - warmup_steps, 1))))
      phase_start_epoch = 0
      if resume_payload is not None:
          resume_phase_names = [name for name, _ in phases]
          if resume_phase_names.index(phase) < resume_phase_names.index(resume_payload["phase"]):
              continue
          if phase == resume_payload["phase"]:
              phase_start_epoch = int(resume_payload["phase_epoch"])
              if phase_start_epoch < phase_epochs:
                  optimizer.load_state_dict(resume_payload["optimizer"])
                  scheduler.load_state_dict(resume_payload["scheduler"])
              else:
                  continue
          resume_payload = None
      best_phase_score = -float("inf")
      if phase_start_epoch and args.resume is not None:
          # args.resume remains available after resume_payload is consumed.
          saved_for_score = torch.load(args.resume, map_location="cpu", weights_only=False)
          best_phase_score = float(saved_for_score.get("best_phase_score", -float("inf")))
      for phase_epoch in range(phase_start_epoch, phase_epochs):
        epoch += 1
        model.train()
        losses = []
        accumulator = ValidMicrobatchAccumulator(
            optimizer, (parameter for parameter in model.parameters() if parameter.requires_grad),
            args.grad_accum)
        for batch_index, batch in enumerate(train_loader):
            fundus, enface = batch["fundus"].to(device), batch["enface"].to(device)
            matrix, valid = batch["gt_matrix"].to(device), batch["valid_mask"].to(device)
            amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                         "float32": torch.float32}[args.amp_dtype]
            amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(fundus, enface, valid)
                pairs = sample_positive_pairs(output["fundus_desc"], output["oct_desc"], matrix,
                                              output["descriptor_valid_mask"], args.k,
                                              fundus_image_size=fundus.shape[-2:], generator=generator)
                if len(pairs.positive) == 0:
                    continue
                negatives = sample_intraimage_negatives(pairs, output["fundus_desc"],
                                                        output["descriptor_valid_mask"], 64, 12.0,
                                                        fundus.shape[-2:])
                loss = dense_infonce(pairs.positive, negatives, args.temperature)
            accumulator.backward(loss)
            if accumulator.ready():
                accumulator.step(scheduler)
                global_step += 1
                if args.smoke_valid_steps and global_step >= args.smoke_valid_steps:
                    smoke = common_checkpoint_payload(
                        model, epoch=epoch, seed=args.seed, fold=args.fold, stats=stats,
                        scale_config=scale_config, encoder=args.encoder, phase=phase,
                        optimizer_provenance={"smoke": True})
                    smoke.update(global_step=global_step, phase_epoch=phase_epoch,
                                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                 rng_state=capture_rng_state(generator), resumable=False)
                    atomic_torch_save(smoke, args.out / "smoke_state.pth")
                    peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
                    print(json.dumps({"smoke_complete": True, "valid_optimizer_steps": global_step,
                                      "peak_cuda_bytes": peak}), flush=True)
                    return
            losses.append(float(loss.detach()))
        if accumulator.step(scheduler):
            global_step += 1
        metrics = representation_metrics(model, val_loader, device, args.k)
        phase_metrics[phase] = metrics
        row = {"epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)), **metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        optimizer_provenance = {"head_lr": args.head_lr,
            "last_block_lr": args.last_block_lr, "layer_decay": args.layer_decay,
            "grad_accum": args.grad_accum, "amp_dtype": args.amp_dtype}
        checkpoint = common_checkpoint_payload(
            model, epoch=epoch, seed=args.seed, fold=args.fold, stats=stats,
            scale_config=scale_config, encoder=args.encoder, phase=phase,
            optimizer_provenance=optimizer_provenance)
        checkpoint.update(recipe=recipe, gate_schema="task2_common_representation_v1",
                          train_case_ids_sha256=train_ids_hash,
                          validation_case_ids_sha256=validation_ids_hash,
                          validation_indices=[int(index) for index in val_idx])
        phase_score = metrics["m2_center_10px"] - metrics["m2_median_px"] / 1000.0
        attach_resume_state(checkpoint, global_step=global_step, phase_epoch=phase_epoch + 1,
                            optimizer=optimizer, scheduler=scheduler, generator=generator,
                            best_phase_score=max(best_phase_score, phase_score),
                            phase_metrics=phase_metrics)
        checkpoint["prerequisite_gate"] = prerequisite_gate
        atomic_torch_save(checkpoint, args.out / "model_1.pth")
        atomic_torch_save(checkpoint, args.out / f"{phase}_latest.pth")
        if phase_score > best_phase_score:
            best_phase_score = phase_score
            atomic_torch_save(checkpoint, args.out / f"{phase}_best.pth")
        print(json.dumps(row), flush=True)
    checkpoint_hash = hashlib.sha256((args.out / "model_1.pth").read_bytes()).hexdigest()
    absolute_gate = (metrics["m1_top1pct"] >= .5 and metrics["m2_center_10px"] >= .5
                     and metrics["m2_median_px"] < 37.12
                     and metrics["shuffle_drop_points"] >= 30.0)
    relative_gate, scenario_gate, control_metrics = True, True, None
    if args.encoder == "dinov2":
        control_payload = json.loads(args.control_results.read_text(encoding="utf-8"))
        control_metrics = control_payload["metrics"]
        relative_gate = (metrics["m2_center_10px"] - control_metrics["m2_center_10px"] >= .05
                         or metrics["m2_median_px"] <= .85 * control_metrics["m2_median_px"])
        shared_scenarios = set(metrics["per_scenario"]) & set(control_metrics["per_scenario"])
        scenario_gate = bool(shared_scenarios) and all(
            metrics["per_scenario"][scenario]["m2_center_10px"]
            >= control_metrics["per_scenario"][scenario]["m2_center_10px"] - .05
            for scenario in shared_scenarios)
    result = {"command": " ".join(__import__("sys").argv), "seed": args.seed,
              "fold": args.fold, "train_indices": list(map(int, train_idx)),
              "validation_indices": list(map(int, val_idx)), "selection_uses_mock": False,
              "coordinate_frame": "task2_canonical_v1",
              "gate_schema": "task2_common_representation_v1",
              "train_case_ids_sha256": train_ids_hash,
              "validation_case_ids_sha256": validation_ids_hash,
              "scale_augmentation": asdict(scale_config), "checkpoint_sha256": checkpoint_hash,
              "metrics": metrics, "phase_metrics": phase_metrics,
              "control_metrics": control_metrics,
              "absolute_gate_pass": absolute_gate, "relative_gate_pass": relative_gate,
              "per_scenario_gate_pass": scenario_gate,
              "timing_gate_pass": metrics["encoder_seconds_per_case"] < 3.0,
              "gate_pass": absolute_gate and relative_gate and scenario_gate
                           and metrics["encoder_seconds_per_case"] < 3.0}
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
