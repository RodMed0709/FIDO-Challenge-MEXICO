#!/usr/bin/env python3
"""T1-93: evaluacion oficial de la cabeza DFL, comparada caso a caso contra
el pipeline geometrico (segmentacion + formula lineal) que usa produccion
hoy (`submissions/r06-fallback-fixed/inference.py`).

Pregunta que responde: ¿la cabeza DFL gana especificamente en los casos
donde la segmentacion NO logra medir (la cola que hoy cae al fallback
constante), o gana/pierde parejo en todos lados?

Metrica: AUC oficial de distancia = media de accuracy@umbral para umbrales
ENTEROS `0..20` sobre el error absoluto en px (`MAX_THRESHOLD_DIST=20`,
`vendor/fido/Codabench Bundle/scoring_program/scoring_keypoints.py`).
Deliberadamente NO se usa `fido.eval_task1.evaluate_distances`: esa funcion
comparte `MAX_THRESHOLD_PX=10` (constante de KEYPOINTS, `fido/geometry.py`)
para distancia tambien, que es un umbral distinto del oficial para
distancia (20) -- ver `experiments/93-t1-dfl/PRE_REGISTRATION.md` para la
nota completa de esta discrepancia, encontrada durante T1-93.

    PYTHONPATH=src python analysis/evaluate_task1_distance_dfl.py \\
      --root data/Task1 --dfl-checkpoint checkpoints/task1_distance_dfl/model_0.pth \\
      --segmentation-checkpoint submissions/r06-fallback-fixed/model_0.pth \\
      --output experiments/93-t1-dfl/EVALUATION.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fido.data.task1 import Task1Dataset, task1_split  # noqa: E402
from fido.models.task1_distance_dfl import Task1DistanceDFLModel  # noqa: E402
from fido.models.unet_bscan_seg import UNet, distance_from_segmentation  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

MAX_THRESHOLD_DIST = 20  # oficial, ver docstring del modulo

# Constantes de produccion vigentes (submissions/r06-fallback-fixed/inference.py
# lineas 62-66), citadas -- no reajustadas aqui.
GEOMETRIC_SCALE_A = 0.9370
GEOMETRIC_SCALE_B = 3.4277
GEOMETRIC_FALLBACK_PX = 203.8


def official_distance_auc(predictions, ground_truth) -> float:
    errors = np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(ground_truth, dtype=np.float64))
    accuracies = [np.mean(errors <= t) for t in range(MAX_THRESHOLD_DIST + 1)]
    return float(np.mean(accuracies))


def mean_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def load_segmentation_checkpoint(path: Path) -> UNet:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["distance"] if isinstance(raw, dict) and "distance" in raw else raw
    state = {k.removeprefix("module."): v for k, v in state.items()}
    base = state["encoders.0.conv1.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("encoders."))
    model = UNet(base_channels=base, depth=depth)
    model.load_state_dict(state)
    model.eval()
    return model


def load_dfl_checkpoint(path: Path) -> Task1DistanceDFLModel:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    model = Task1DistanceDFLModel(
        reg_max=raw["reg_max"], d_min=raw["d_min"], d_max=raw["d_max"],
        base_channels=raw["base_channels"], depth=raw["depth"], head_hidden=raw["head_hidden"],
    )
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model


def _auc_and_mean(errors: np.ndarray) -> tuple[float, float]:
    if errors.size == 0:
        return float("nan"), float("nan")
    accuracies = [np.mean(errors <= t) for t in range(MAX_THRESHOLD_DIST + 1)]
    return float(np.mean(accuracies)), float(errors.mean())


def _row(label: str, errors: np.ndarray) -> str:
    auc, mean_error = _auc_and_mean(errors)
    return f"| {label} | {errors.size:,} | {auc:.4f} | {mean_error:.2f} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case-cache", type=Path, default=None)
    parser.add_argument("--dfl-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments" / "93-t1-dfl" / "EVALUATION.md")
    args = parser.parse_args()

    device = torch.device(args.device)
    dfl_model = load_dfl_checkpoint(args.dfl_checkpoint).to(device)
    seg_model = load_segmentation_checkpoint(args.segmentation_checkpoint).to(device)

    dataset = Task1Dataset(args.root, load_fundus=False, load_bscan=True,
                            cache_path=args.case_cache)
    _, val_split = task1_split(dataset.cases, n_splits=args.n_folds, fold=args.fold, seed=args.seed)
    val_loader = DataLoader(Subset(dataset, val_split.indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    ground_truth, scenarios = [], []
    dfl_pred, geometric_pred, geometric_measured = [], [], []
    started = time.time()

    with torch.no_grad():
        for count, batch in enumerate(val_loader, 1):
            bscan = batch["bscan"].to(device)
            has_oct = batch["has_oct"].cpu().numpy()
            distances = batch["distance"].cpu().numpy()
            B, n_slices, H, W = bscan.shape

            dfl_out = dfl_model(bscan)["distance"].cpu().numpy()

            seg_logits = seg_model(bscan.reshape(B * n_slices, 1, H, W)).reshape(B, n_slices, 3, H, W)

            for i in range(B):
                ground_truth.append(float(distances[i]))
                scenarios.append(batch["scenario"][i])
                dfl_pred.append(float(dfl_out[i]) if has_oct[i] else float("nan"))

                gaps = []
                if has_oct[i]:
                    for s in range(n_slices):
                        gap = distance_from_segmentation(seg_logits[i:i + 1, s]).item()
                        if np.isfinite(gap):
                            gaps.append(gap)
                gap = mean_finite(gaps)
                measured = np.isfinite(gap)
                geometric_measured.append(bool(measured))
                geometric_pred.append(GEOMETRIC_SCALE_A * gap + GEOMETRIC_SCALE_B if measured
                                      else GEOMETRIC_FALLBACK_PX)
                # DFL no depende de segmentacion: si hay OCT, siempre predice.
                # Si NO hay OCT (10% del dataset), no hay imagen que darle --
                # usamos el mismo fallback geometrico para no penalizar a DFL
                # por un problema que no es el que ataca (ver PRE_REGISTRATION.md).
                if not has_oct[i]:
                    dfl_pred[-1] = GEOMETRIC_FALLBACK_PX

            if count % 50 == 0:
                print(f"[{time.time()-started:6.1f}s] {count * args.batch_size:,} casos procesados",
                      flush=True)

    ground_truth = np.asarray(ground_truth)
    dfl_pred = np.asarray(dfl_pred)
    geometric_pred = np.asarray(geometric_pred)
    geometric_measured = np.asarray(geometric_measured)

    dfl_errors = np.abs(dfl_pred - ground_truth)
    geometric_errors = np.abs(geometric_pred - ground_truth)

    measured_mask = geometric_measured
    unmeasured_mask = ~geometric_measured

    lines = [
        "# T1-93 -- evaluacion oficial: cabeza DFL vs pipeline geometrico",
        "",
        f"**DFL checkpoint:** `{args.dfl_checkpoint}`  ",
        f"**Segmentation checkpoint (referencia de coverage):** `{args.segmentation_checkpoint}`  ",
        f"**Split:** GroupKFold {args.n_folds}, fold {args.fold}, seed {args.seed}  ",
        f"**n validation:** {len(ground_truth):,}  ",
        f"**Umbral oficial:** enteros 0..{MAX_THRESHOLD_DIST} "
        f"(`MAX_THRESHOLD_DIST`, `scoring_keypoints.py`)  ",
        f"**Tiempo:** {time.time() - started:.1f}s  ",
        "",
        "## Pooled (todos los casos de validation)",
        "",
        "| metodo | n | distance_auc (umbral 0..20) | mean_error (px) |",
        "|---|---:|---:|---:|",
        _row("DFL", dfl_errors),
        _row("Geometrico (produccion)", geometric_errors),
        "",
        "## Desglose: casos donde la segmentacion SI mide vs NO mide",
        "",
        f"Coverage geometrico en este split: {measured_mask.mean():.4f} "
        f"({measured_mask.sum():,} de {len(measured_mask):,})  ",
        "",
        "| subset | metodo | n | distance_auc (umbral 0..20) | mean_error (px) |",
        "|---|---|---:|---:|---:|",
        "| segmentacion mide " + _row("DFL", dfl_errors[measured_mask]),
        "| segmentacion mide " + _row("Geometrico", geometric_errors[measured_mask]),
        "| segmentacion NO mide " + _row("DFL", dfl_errors[unmeasured_mask]),
        "| segmentacion NO mide " + _row("Geometrico", geometric_errors[unmeasured_mask]),
        "",
        "## Reproduccion",
        "",
        "```bash",
        "PYTHONPATH=src python analysis/evaluate_task1_distance_dfl.py \\",
        f"  --root {args.root} --dfl-checkpoint {args.dfl_checkpoint} \\",
        f"  --segmentation-checkpoint {args.segmentation_checkpoint} \\",
        f"  --n-folds {args.n_folds} --fold {args.fold} --seed {args.seed}",
        "```",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
