"""T1-93: entrenamiento de la cabeza de distancia por bins (DFL) para Task 1.

Ver `src/fido/models/task1_distance_dfl.py` para el porque de esta cabeza
(complementa/reemplaza el pipeline geometrico de `train_task1_unet.py` en
los casos donde la segmentacion NO encuentra clase instrumento o ILM -- ver
`experiments/93-t1-dfl/NAN_FRACTION_MEASUREMENT.md`) y
`experiments/93-t1-dfl/PRE_REGISTRATION.md` para la hipotesis, el rango
`[d_min, d_max]`/`reg_max` elegido y su justificacion empirica.

Split: GroupKFold POR ESCENARIO via `fido.data.task1.task1_split` --
obligatorio, mismo criterio que `train_task1_unet.py` y `train_task1_keypoint.py`.

Entrena SOLO sobre casos con `has_oct=True` (el 10% sin OCT es un problema
distinto -- ya lo resuelve el fallback constante de `inference.py`, entrenar
la cabeza sobre entrada en cero no aporta señal real). El fallback para esos
casos se calibra aparte (mediana de las distancias de train, guardada en el
mismo formato `DistanceCalibration` que ya usa `train_task1_unet.py`, con
`scale=1.0, offset=0.0` porque la cabeza DFL no necesita reescalado lineal
posterior -- ya decodifica directo a px).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fido.data.task1 import Task1Dataset, task1_split
from fido.models.task1_distance_dfl import (
    Task1DistanceDFLModel,
    dfl_loss,
    load_pretrained_segmentation_encoder,
)
from fido.train.train_task1_unet import (
    DistanceCalibration,
    save_distance_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/task1_distance_dfl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--case-cache", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None)

    # Hiperparametros de la cabeza DFL -- SIN default "magico" copiado del
    # paper (`-1mm..6mm`, otras unidades/sensor): el rango se mide sobre los
    # 61,691 casos reales de Task 1 en
    # experiments/93-t1-dfl/DISTANCE_TARGET_HISTOGRAM.md -- min=6.7px,
    # max=1178.8px, mediana=216.7px, SIN casos negativos (a diferencia del
    # paper, que reserva rango negativo para inyeccion subretiniana; ese
    # regimen no aparece en nuestros datos). d_min=0 (ningun caso por debajo),
    # d_max=1200 (por encima del maximo observado, deja margen sin recortar
    # casi ningun caso real). reg_max=128 -> ancho de bin = 1200/128 = 9.375px,
    # mas fino que el umbral oficial de distancia (0..20px) para que la
    # esperanza ponderada tenga margen de dar sub-bin. El CLI siempre puede
    # overridearlos.
    parser.add_argument("--reg-max", type=int, default=128)
    parser.add_argument("--d-min", type=float, default=0.0)
    parser.add_argument("--d-max", type=float, default=1200.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--pretrained-segmentation-checkpoint", type=Path, default=None,
                        help="Opcional: state_dict de unet_bscan_seg.UNet (p.ej. "
                             "raw['distance'] de submissions/r06-fallback-fixed/model_0.pth) "
                             "para inicializar el encoder por transferencia. Requiere "
                             "--base-channels/--depth identicos a ese checkpoint.")
    return parser.parse_args()


def _median_finite(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        raise ValueError("no finite distances available to compute fallback")
    return float(np.median(finite))


def fit_dfl_fallback(dataloader: DataLoader) -> DistanceCalibration:
    """Fallback constante para casos SIN OCT (`has_oct=False`): mediana de
    las distancias GT de train, exclusivamente sobre train (mismo criterio
    anti-leakage que `fit_distance_calibration` en `train_task1_unet.py`)."""
    distances, case_ids = [], []
    for batch in dataloader:
        for i in range(len(batch["distance"])):
            distances.append(float(batch["distance"][i]))
            case_ids.append(f"{batch['scenario'][i]}/{batch['frame_id'][i]}")
    fallback = _median_finite(distances)
    return DistanceCalibration(scale=1.0, offset=0.0, fallback=fallback,
                               fit_case_ids=tuple(case_ids))


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
            fallback: float) -> dict:
    """Evalua sobre TODOS los casos de `dataloader` (incluye `has_oct=False`,
    que recibe el `fallback` constante). Devuelve las listas crudas -- el
    AUC oficial (umbral 0..20) se calcula en `analysis/evaluate_task1_distance_dfl.py`,
    no aqui, para no duplicar la definicion del scorer en dos sitios."""
    model.eval()
    predictions, ground_truth, scenarios, case_ids, has_oct_flags = [], [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            bscan = batch["bscan"].to(device)
            has_oct = batch["has_oct"].cpu().numpy()
            distances = batch["distance"].cpu().numpy()
            out = model(bscan)
            batch_pred = out["distance"].cpu().numpy()
            for i in range(len(distances)):
                pred = float(batch_pred[i]) if has_oct[i] else fallback
                predictions.append(pred)
                ground_truth.append(float(distances[i]))
                scenarios.append(batch["scenario"][i])
                case_ids.append(f"{batch['scenario'][i]}/{batch['frame_id'][i]}")
                has_oct_flags.append(bool(has_oct[i]))
    return {
        "predictions": predictions, "ground_truth": ground_truth,
        "scenarios": scenarios, "case_ids": case_ids, "has_oct": has_oct_flags,
    }


def official_distance_auc(predictions, ground_truth, max_threshold: int = 20) -> float:
    """AUC oficial: media de accuracy@umbral para umbrales enteros
    `0..max_threshold` sobre `|pred - gt|`. Replica
    `vendor/fido/Codabench Bundle/scoring_program/scoring_keypoints.py::auc_from_errors`
    con `max_threshold=MAX_THRESHOLD_DIST=20` -- NO reutiliza
    `fido.eval_task1.evaluate_distances` porque esa funcion usa
    `MAX_THRESHOLD_PX=10` (constante de KEYPOINTS, geometry.py) para
    distancia tambien, que no es el umbral oficial de distancia (20). Ver
    la nota en `experiments/93-t1-dfl/PRE_REGISTRATION.md`."""
    errors = np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(ground_truth, dtype=np.float64))
    accuracies = [np.mean(errors <= threshold) for threshold in range(max_threshold + 1)]
    return float(np.mean(accuracies))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    try:
        from fido import ledger as _ledger
        _inicio = _ledger.ahora()
    except Exception as _exc:  # noqa: BLE001
        _ledger, _inicio = None, None
        print(f"[ledger] AVISO: no disponible ({_exc}); la corrida no se registrará")

    dataset = Task1Dataset(args.root, load_fundus=False, load_bscan=True,
                            cache_path=args.case_cache)
    unique_scenarios = {c["scenario"] for c in dataset.cases}
    if len(unique_scenarios) < args.n_folds:
        print(
            f"AVISO: solo {len(unique_scenarios)} escenarios únicos < n_folds={args.n_folds} "
            "-> usando TODO el dataset para train y val (solo smoke test, no válido para elegir hiperparámetros)."
        )
        train_idx = list(range(len(dataset)))
        val_idx = list(range(len(dataset)))
    else:
        train_split, val_split = task1_split(
            dataset.cases, n_splits=args.n_folds, fold=args.fold, seed=args.seed
        )
        train_idx, val_idx = train_split.indices, val_split.indices

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model = Task1DistanceDFLModel(
        reg_max=args.reg_max, d_min=args.d_min, d_max=args.d_max,
        base_channels=args.base_channels, depth=args.depth, head_hidden=args.head_hidden,
    ).to(device)

    if args.pretrained_segmentation_checkpoint is not None:
        raw = torch.load(args.pretrained_segmentation_checkpoint, map_location="cpu",
                         weights_only=False)
        state = raw["distance"] if isinstance(raw, dict) and "distance" in raw else raw
        state = {k.removeprefix("module."): v for k, v in state.items()}
        n_copied = load_pretrained_segmentation_encoder(model.encoder, state)
        print(f"Encoder inicializado desde {args.pretrained_segmentation_checkpoint}: "
              f"{n_copied} tensores copiados de {len(model.encoder.state_dict())}")
        if n_copied == 0:
            print("AVISO: 0 tensores copiados -- revisa --base-channels/--depth contra el checkpoint")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    fallback_calibration = fit_dfl_fallback(train_loader)
    val_ids = {f"{dataset.cases[int(i)]['scenario']}/{dataset.cases[int(i)]['frame_id']}"
               for i in val_idx}
    if set(fallback_calibration.fit_case_ids) & val_ids:
        raise RuntimeError("DFL fallback calibration leaked validation case IDs")
    print(f"Fallback (mediana train, casos sin OCT): {fallback_calibration.fallback:.3f}px "
          f"n_fit={len(fallback_calibration.fit_case_ids)}")
    args.out.mkdir(parents=True, exist_ok=True)
    save_distance_calibration(args.out / "distance_calibration.json", fallback_calibration)

    best_auc = -1.0
    best_mean_error = float("inf")
    best_epoch = -1
    best_n_val = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            has_oct = batch["has_oct"]
            if not has_oct.any():
                continue
            bscan = batch["bscan"][has_oct].to(device)
            distance = batch["distance"][has_oct].to(device)

            optimizer.zero_grad()
            out = model(bscan)
            loss = dfl_loss(out["logits"], distance, model.head)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        if not train_losses:
            print(f"[epoch {epoch}] sin batches con OCT, se saltó la época entera")
            continue

        avg_loss = float(np.mean(train_losses))
        print(f"[epoch {epoch}] train_dfl_loss={avg_loss:.4f}")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            results = evaluate(model, val_loader, device, fallback_calibration.fallback)
            n_valid = len(results["predictions"])
            if n_valid == 0:
                print(f"[epoch {epoch}]   sin casos de validation")
                continue
            errors = np.abs(np.asarray(results["predictions"]) - np.asarray(results["ground_truth"]))
            mean_error = float(errors.mean())
            val_auc = official_distance_auc(results["predictions"], results["ground_truth"])
            coverage = float(np.mean(results["has_oct"]))
            print(
                f"[epoch {epoch}]   val_distance_mae={mean_error:.2f}px  "
                f"val_distance_auc={val_auc:.4f}  coverage_oct={coverage:.4f}  n={n_valid}"
            )

            is_better = val_auc > best_auc or (val_auc == best_auc and mean_error < best_mean_error)
            if is_better:
                best_auc = val_auc
                best_mean_error = mean_error
                best_epoch = epoch
                best_n_val = n_valid
                torch.save({"state_dict": model.state_dict(),
                           "reg_max": args.reg_max, "d_min": args.d_min, "d_max": args.d_max,
                           "base_channels": args.base_channels, "depth": args.depth,
                           "head_hidden": args.head_hidden},
                          args.out / "model_0.pth")
                print(f"  -> nuevo mejor (auc={best_auc:.4f} mae={mean_error:.2f}px) "
                      f"en epoch {epoch}, checkpoint guardado")

    if best_epoch == -1:
        print("No se pudo evaluar AUC en ninguna época (sin casos de val).")
    else:
        print(f"Mejor AUC: {best_auc:.4f} en época {best_epoch}")

    if _ledger is not None:
        _ledger.record_run_safe(
            id=_ledger.id_de_corrida("93-t1-dfl", _inicio),
            titulo="T1 distancia: cabeza DFL (bins) sobre encoder de B-scan",
            peldano="93-t1-dfl",
            script="fido.train.train_task1_distance_dfl",
            task=1,
            estado="completada" if best_epoch != -1 else "abortada",
            metricas=({"val_distance_auc": float(best_auc),
                       "val_distance_mae_px": float(best_mean_error)}
                      if best_epoch != -1 else {}),
            epoca=best_epoch if best_epoch != -1 else None,
            epocas_planeadas=args.epochs,
            n_val=best_n_val,
            fold=args.fold,
            n_folds=args.n_folds,
            checkpoint=str(args.out / "model_0.pth"),
            semilla=args.seed,
            inicio=_inicio,
            notas=("Registrado automáticamente al terminar. reg_max="
                   f"{args.reg_max} d_min={args.d_min} d_max={args.d_max}. Las métricas son "
                   "las de la MEJOR época, no las de la última."),
        )


if __name__ == "__main__":
    main()
