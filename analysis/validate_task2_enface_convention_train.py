#!/usr/bin/env python3
"""Validate the frozen Task 2 en-face convention on one training fold only."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from analysis.verify_task2_enface_convention import (  # noqa: E402
    RADII, VARIANTS, aggregate, measure_case, ratio, table,
)
from fido.data.common import TASK2_ENFACE_CONVENTION, group_kfold_indices  # noqa: E402
from fido.data.common import TASK2_CANONICAL_TO_NATIVE  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402
from fido.eval_task2 import require_task2_training_root  # noqa: E402


def select_train_cases(cases: list[dict], n_folds: int, fold: int, seed: int):
    groups = [case["scenario"] for case in cases]
    splits = list(group_kfold_indices(groups, n_folds, seed))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0,{len(splits) - 1}]")
    train_idx, val_idx = splits[fold]
    if set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[val_idx]):
        raise RuntimeError("scenario leakage in GroupKFold")
    return [cases[int(index)] for index in train_idx], train_idx, val_idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    root = require_task2_training_root(args.root)
    cases = find_task2_cases(root)
    train_cases, train_idx, val_idx = select_train_cases(cases, args.n_folds, args.fold, args.seed)

    ordered = [None] * len(train_cases)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(measure_case, case, index, args.seed): index
                   for index, case in enumerate(train_cases)}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index = futures[future]
            ordered[index] = future.result()
            if completed % 25 == 0 or completed == len(train_cases):
                print(f"[{completed}/{len(train_cases)}] train-only convention oracle", flush=True)
    valid = [result for result in ordered if result is not None and not result["error"]]
    instrument = aggregate(valid, "c")
    vessel = aggregate(valid, "a")
    expected = instrument[TASK2_ENFACE_CONVENTION]
    identity = instrument["identity__keep_u__keep_v"]
    radii = RADII[1:3]
    expected_ratios = [ratio(*expected["rates"][radius]) for radius in radii]
    identity_ratios = [ratio(*identity["rates"][radius]) for radius in radii]
    gate_pass = (expected["n"] >= 100 and all(value >= 2.0 for value in expected_ratios)
                 and all(win >= 1.5 * control for win, control in zip(expected_ratios, identity_ratios)))
    case_ids = [f"{case['scenario']}/{case['frame_id']}" for case in train_cases]
    validation_case_ids = [f"{cases[int(index)]['scenario']}/{cases[int(index)]['frame_id']}"
                           for index in val_idx]
    payload = {
        "root": str(root), "seed": args.seed, "n_folds": args.n_folds, "fold": args.fold,
        "selection_uses_mock": False, "expected_convention": TASK2_ENFACE_CONVENTION,
        "gate_schema": "task2_enface_convention_gate_v1",
        "canonical_to_native_matrix": TASK2_CANONICAL_TO_NATIVE.tolist(),
        "matrix_contract": "M_canonical = M_native @ C",
        "train_indices": [int(i) for i in train_idx],
        "validation_indices": [int(i) for i in val_idx],
        "train_case_ids_sha256": hashlib.sha256("\n".join(case_ids).encode()).hexdigest(),
        "validation_case_ids_sha256": hashlib.sha256(
            "\n".join(validation_case_ids).encode()).hexdigest(),
        "requested_train_cases": len(train_cases), "valid_cases": len(valid),
        "instrument_points": expected["n"],
        "radii": list(radii), "expected_ratios": expected_ratios,
        "identity_ratios": identity_ratios,
        "gate": {"min_points": 100, "expected_ratio_min": 2.0,
                 "relative_to_identity_min": 1.5}, "gate_pass": bool(gate_pass),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# T2-83 train-only en-face convention oracle", "",
             f"Expected: `{TASK2_ENFACE_CONVENTION}`; fold={args.fold}/{args.n_folds}; seed={args.seed}.",
             f"Train cases={len(train_cases)}; valid={len(valid)}; Mock used=false.", ""]
    lines += table("Instrument", instrument) + [""] + table("Vasculature", vessel)
    lines += ["", "## Gate", "", f"`gate_pass={str(gate_pass).lower()}`", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gate_pass": gate_pass, "valid": len(valid),
                      "instrument_points": expected["n"]}), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
