"""Entrenamiento de `Task2DPCNModel` (T2-92, ver ATTACK_LADDER.md) --
correlación de fase log-polar diferenciable (DPCN/DPCN++) con localización
gruesa previa, dentro del mismo protocolo que el resto de Task 2:
GroupKFold por escenario, augmentación de escala exacta derivada solo de
train, y control OCT-shuffle OBLIGATORIO reportado junto a la métrica
principal en cada validación.

Sigue la misma estructura que `train_task2_baseline.py` a propósito
(consistencia con el resto de la escalera), con dos añadidos específicos de
este peldaño:

1. `--oct-shuffle-check`: en cada validación, además del AUC/mean_error
   normales, se recorre el mismo val_loader con el en-face BARAJADO dentro
   del batch (fundus y gt_matrix quedan intactos). Si el AUC con
   OCT-shuffle NO se degrada respecto al AUC normal, el modelo no está
   usando el OCT y el número "bueno" es basura aunque el AUC principal se
   vea alto -- exactamente el control que salvó al proyecto en T2-70/T2-80
   (ver ATTACK_LADDER.md).
2. Pérdida auxiliar sobre el centro de la etapa GRUESA
   (`coarse_center_loss`): sin ella, el único gradiente que llega a
   `_coarse_localize` pasa por TODA la cadena FFT/log-polar/ajuste
   cerrado -- una señal indirecta y potencialmente débil al arrancar.
   Supervisar el centro grueso directamente con el centro del GT (mismo
   punto que usa la correlación cruzada de `task2_baseline.py`, la esquina
   uv=(0.5,0.5) proyectada) da a la localización gruesa una señal de
   entrenamiento propia desde la época 1.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fido.data.common import group_kfold_indices
from fido.data.task2 import Task2Dataset, Task2TransformSubset, load_task2_scales
from fido.data.task2_transforms import Task2GeometricAugment, derive_scale_augmentation
from fido.geometry import corner_auc, corner_error, huber_saturated
from fido.models.task2_dpcn import Task2DPCNModel
from fido.train.train_task2_common import select_group_fold


def shuffle_enface_batch(enface: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Control OCT-shuffle: permuta el en-face entre muestras del batch
    (fundus y gt_matrix quedan en su índice original). Con batch=1 no hay
    nada que barajar -- el llamador debe usar un batch de validación >=2
    para que el control sea informativo."""
    batch = enface.shape[0]
    if batch < 2:
        raise ValueError("oct-shuffle requires a validation batch size >= 2")
    if generator is not None:
        permutation = torch.randperm(batch, generator=generator)
    else:
        permutation = torch.randperm(batch)
    # Evita que la permutación identidad (posible al azar) vuelva el
    # control silenciosamente inerte: si sale la identidad, se rota uno.
    if torch.equal(permutation, torch.arange(batch)):
        permutation = torch.roll(permutation, shifts=1)
    return enface[permutation]


def compute_loss(pred: dict, batch: dict, fundus_size: int) -> tuple[torch.Tensor, dict]:
    gt_matrix = batch["gt_matrix"]

    # Pérdida principal: exactamente la métrica del challenge (corner AUC),
    # vía la variante robusta y saturada -- mismo razonamiento que
    # `train_task2_baseline.py`: un fallo catastrófico no debe competir por
    # gradiente con casos ya recuperables.
    errors = corner_error(pred["pred_matrix"], gt_matrix)
    corner_loss = huber_saturated(errors).mean()

    # Auxiliar: centro GT (esquina uv=(0.5,0.5) proyectada) contra el
    # centro de la etapa gruesa -- ver docstring del módulo. Normalizado
    # por el tamaño de imagen, mismo motivo que `coord_l1` en
    # `train_task2_baseline.py` (evitar que domine `clip_grad_norm_`).
    gt_center = gt_matrix[:, :2, :2] @ torch.full(
        (2,), 0.5, device=gt_matrix.device, dtype=gt_matrix.dtype
    ) + gt_matrix[:, :2, 2]
    coarse_center_loss = F.l1_loss(pred["coarse_center"] / fundus_size, gt_center / fundus_size)

    loss_total = corner_loss + 5.0 * coarse_center_loss
    return loss_total, {
        "corner": corner_loss.item(),
        "coarse_center": coarse_center_loss.item(),
        "mean_corner_error_px": errors.mean().item(),
    }


def _run_validation(model, loader, device, *, oct_shuffle: bool, shuffle_seed: int):
    model.eval()
    errors, shuffled_errors = [], []
    generator = torch.Generator().manual_seed(shuffle_seed)
    with torch.no_grad():
        for batch in loader:
            device_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = model(device_batch["fundus"], device_batch["enface"],
                        valid_mask=device_batch.get("valid_mask"))
            errors.extend(corner_error(pred["pred_matrix"], device_batch["gt_matrix"]).cpu().numpy().tolist())

            if oct_shuffle and device_batch["enface"].shape[0] >= 2:
                shuffled_enface = shuffle_enface_batch(device_batch["enface"], generator=generator)
                shuffled_pred = model(device_batch["fundus"], shuffled_enface,
                                      valid_mask=device_batch.get("valid_mask"))
                shuffled_errors.extend(
                    corner_error(shuffled_pred["pred_matrix"], device_batch["gt_matrix"]).cpu().numpy().tolist()
                )
    result = {"errors": np.array(errors)}
    if oct_shuffle and shuffled_errors:
        result["shuffled_errors"] = np.array(shuffled_errors)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/task2_dpcn"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--enface-cache", type=Path, default=None,
                         help="Ver infra/precompute_task2_enface.py -- evita leer los 128 PNG por caso.")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--scale-augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Augmentación de escala exacta (T2-81); default ON en este peldaño porque "
                             "el rango de escala de Mock/test NO se solapa con train (ver CLAUDE.md).")
    parser.add_argument("--scale-extrapolation", type=float, default=0.25)
    parser.add_argument("--oct-shuffle-check", action=argparse.BooleanOptionalAction, default=True,
                        help="Control OBLIGATORIO (ver PRE_REGISTRATION.md T2-92): repite cada "
                             "validación con el en-face barajado dentro del batch.")
    # Hiperparámetros de arquitectura -- defaults densos/potentes a
    # propósito (encargo: medir la familia DPCN sin escatimar capacidad).
    parser.add_argument("--window-size", type=float, default=512.0,
                         help="Lado (px, fundus 1024) de la ventana candidata recortada antes del "
                              "núcleo Fourier-Mellin. Cubre con margen holgado el rango de escala "
                              "aumentado (train 126-188px, Mock 213-226px).")
    parser.add_argument("--working-size", type=int, default=256)
    parser.add_argument("--fine-feat-size", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        from fido import ledger as _ledger
        _inicio = _ledger.ahora()
    except Exception as _exc:  # noqa: BLE001
        _ledger, _inicio = None, None
        print(f"[ledger] AVISO: no disponible ({_exc}); la corrida no se registrará")

    dataset = Task2Dataset(args.root, include_vessel_enface=False, include_vessel_mask=False,
                            enface_cache_dir=args.enface_cache)
    groups = [c["scenario"] for c in dataset.cases]
    unique_scenarios = set(groups)

    if len(unique_scenarios) < args.n_folds:
        print(f"[AVISO] Solo hay {len(unique_scenarios)} escenario(s) - usando train=val completo. "
              "Valido SOLO para un smoke test de overfit, nunca para elegir hiperparametros.")
        train_idx = list(range(len(dataset)))
        val_idx = list(range(len(dataset)))
    else:
        train_idx, val_idx = select_group_fold(groups, args.n_folds, args.fold, seed=args.seed)

    train_transform = None
    scale_config = None
    if args.scale_augment:
        scale_config = derive_scale_augmentation(
            load_task2_scales(dataset.cases), train_idx, args.scale_extrapolation)
        train_transform = Task2GeometricAugment(scale_config.factor_min, scale_config.factor_max,
                                                seed=args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "scale_augmentation.json").write_text(json.dumps(asdict(scale_config), indent=2),
                                                           encoding="utf-8")
        print(f"[scale-augment] factor=[{scale_config.factor_min:.5f}, {scale_config.factor_max:.5f}] "
              f"source_rows={len(scale_config.source_indices)}")

    train_dataset = Task2TransformSubset(dataset, train_idx, train_transform)
    val_dataset = Task2TransformSubset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=max(args.batch_size, 2), shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                            prefetch_factor=4 if args.num_workers > 0 else None)

    model = Task2DPCNModel(window_size=args.window_size, working_size=args.working_size,
                           fine_feat_size=args.fine_feat_size)
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Task2DPCNModel: {n_params:,} parametros")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    best_mean_error = float("inf")
    best_epoch = -1
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "corner": 0.0, "coarse_center": 0.0, "mean_corner_error_px": 0.0}
        n_batches = 0

        for batch in train_loader:
            device_batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            pred = model(device_batch["fundus"], device_batch["enface"],
                        valid_mask=device_batch.get("valid_mask"))
            loss, sub_losses = compute_loss(pred, device_batch, fundus_size=1024)
            loss.backward()
            # Mismo clip que task2_baseline.py: el pico inicial de las
            # correlaciones cruzadas (aquí, ademas, la superficie de
            # correlacion de fase) puede dar gradientes enormes en las
            # primeras epocas.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals["loss"] += loss.item()
            for key, value in sub_losses.items():
                totals[key] += value
            n_batches += 1

        averaged = {key: value / max(n_batches, 1) for key, value in totals.items()}
        print(f"[epoch {epoch}] train_loss={averaged['loss']:.4f} corner={averaged['corner']:.3f} "
              f"coarse_center={averaged['coarse_center']:.4f} "
              f"mean_corner_error_px={averaged['mean_corner_error_px']:.2f}")

        is_last_epoch = epoch == args.epochs
        if epoch % args.val_every == 0 or is_last_epoch:
            result = _run_validation(model, val_loader, args.device,
                                     oct_shuffle=args.oct_shuffle_check, shuffle_seed=args.seed + epoch)
            val_errors = result["errors"]
            auc = corner_auc(val_errors)
            mean_error = val_errors.mean()
            log_line = f"[epoch {epoch}]   val_corner_auc={auc:.4f} val_mean_error={mean_error:.2f}px n={len(val_errors)}"
            if "shuffled_errors" in result:
                shuffled_auc = corner_auc(result["shuffled_errors"])
                shuffled_mean_error = result["shuffled_errors"].mean()
                log_line += (f"  |  OCT-SHUFFLE: auc={shuffled_auc:.4f} "
                            f"mean_error={shuffled_mean_error:.2f}px delta_auc={auc - shuffled_auc:+.4f}")
            print(log_line)

            is_better = auc > best_auc or (auc == best_auc and mean_error < best_mean_error)
            if is_better:
                best_auc, best_mean_error, best_epoch = auc, mean_error, epoch
                torch.save(model.state_dict(), args.out / "model_1.pth")
                print(f"  -> nuevo mejor (auc={auc:.4f} mean_error={mean_error:.2f}px), "
                      f"guardado en {args.out / 'model_1.pth'}")

    print(f"Mejor AUC: {best_auc:.4f} en epoca {best_epoch}")

    if _ledger is not None:
        _ledger.record_run_safe(
            id=_ledger.id_de_corrida("p92-t2-dpcn", _inicio),
            titulo="T2 registracion: DPCN log-polar + correlacion de fase diferenciable",
            peldano="92-t2-dpcn",
            script="fido.train.train_task2_dpcn",
            task=2,
            estado="completada" if best_epoch != -1 else "abortada",
            metricas=({"val_corner_auc": float(best_auc), "val_mean_error_px": float(best_mean_error)}
                      if best_epoch != -1 else {}),
            epoca=best_epoch if best_epoch != -1 else None,
            epocas_planeadas=args.epochs,
            n_val=len(val_dataset),
            fold=args.fold,
            n_folds=args.n_folds,
            checkpoint=str(args.out / "model_1.pth"),
            semilla=args.seed,
            inicio=_inicio,
            notas="Registrado automaticamente al terminar. Metricas de la MEJOR epoca (val_corner_auc "
                  "normal, NO el control OCT-shuffle -- ver log de la epoca para el delta).",
        )


if __name__ == "__main__":
    main()
