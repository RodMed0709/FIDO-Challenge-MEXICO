from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fido.data.task1 import Task1Dataset, task1_split
from fido.data.task1_augment import AugmentedTask1Dataset, get_task1_augment_config
from fido.eval_task1 import evaluate_keypoints, write_evaluation_outputs
from fido.models.task1_keypoint import Task1KeypointModel
from fido.models.task1_keypoint_dinov2 import Task1KeypointDinoV2Model
from fido.models.task1_keypoint_resnet import Task1KeypointResNet18FPN
from fido.heatmap_decode import gaussian_heatmap_target
from fido.task1_checkpoint import make_task1_keypoint_payload


def compute_loss(
    pred: dict,
    batch: dict,
    fundus_size: int = 1024,
    heatmap_sigma: float = 2.0,
    heatmap_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    gt_keypoint = batch['keypoint']  # (B, 2), orden (x, y), en píxeles del fundus
    # L1 sin saturar sobre (x,y): necesaria para que el modelo reciba señal de gradiente
    # al arrancar desde cero con errores de cientos de píxeles. NO usar aquí ninguna
    # pérdida tipo huber_saturated (satura por encima de 15px, gradiente cero al empezar).
    coord_l1 = F.l1_loss(pred['keypoint'], gt_keypoint)

    heat_h, heat_w = pred['heatmap_logits'].shape[-2:]
    # heatmap cuadrado (fundus cuadrado), un solo stride sirve para x e y
    stride = fundus_size / heat_w
    # gaussian_heatmap_target espera coordenadas en unidades de grilla del heatmap,
    # no en píxeles reales del fundus
    center_heatmap_units = gt_keypoint / stride  # (B,2)

    target_heatmap = gaussian_heatmap_target(
        center_heatmap_units, heat_h, heat_w, sigma=heatmap_sigma,
        device=pred['heatmap_logits'].device, dtype=pred['heatmap_logits'].dtype
    ).unsqueeze(1)  # (B,1,heat_h,heat_w), coincide con la forma de heatmap_logits

    heatmap_bce = F.binary_cross_entropy_with_logits(pred['heatmap_logits'], target_heatmap)
    loss_total = coord_l1 + heatmap_weight * heatmap_bce
    return loss_total, {'coord_l1': coord_l1.item(), 'heatmap_bce': heatmap_bce.item()}


def evaluate(model, dataloader, device,
             predictions_out: Path | None = None) -> tuple[float, float, int]:
    model.eval()
    predictions = []
    ground_truth = []
    scenarios = []
    case_ids = []
    with torch.no_grad():
        for batch in dataloader:
            fundus = batch['fundus'].to(device)
            gt_keypoint = batch['keypoint'].to(device)
            pred = model(fundus)
            predictions.extend(pred['keypoint'].cpu().numpy())
            ground_truth.extend(gt_keypoint.cpu().numpy())
            scenarios.extend(batch['scenario'])
            case_ids.extend(f"{scenario}/{frame_id}" for scenario, frame_id in
                            zip(batch['scenario'], batch['frame_id']))
    if len(predictions) == 0:
        return (float('nan'), float('nan'), 0)
    metrics = evaluate_keypoints(predictions, ground_truth, scenarios)
    if predictions_out is not None:
        write_evaluation_outputs(predictions_out, metrics, case_ids)
    return (metrics['auc'], metrics['mean_error'], metrics['n'])


def build_optimizer_param_groups(model, lr: float,
                                 backbone_lr_multiplier: float = 0.1) -> list[dict]:
    """Separa backbone preentrenado y decoder sin duplicar parámetros."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    backbone_params = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    decoder_params = [parameter for parameter in model.parameters()
                      if parameter.requires_grad and id(parameter) not in backbone_ids]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * backbone_lr_multiplier})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr})
    return groups


def accumulation_window_size(batch_index: int, n_batches: int, steps: int) -> int:
    if steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return min(steps, n_batches - (batch_index // steps) * steps)


def should_optimizer_step(batch_index: int, n_batches: int, steps: int) -> bool:
    accumulation_window_size(batch_index, n_batches, steps)
    return (batch_index + 1) % steps == 0 or batch_index + 1 == n_batches


def scale_loss_for_accumulation(loss: torch.Tensor, batch_index: int,
                                n_batches: int, steps: int) -> torch.Tensor:
    return loss / accumulation_window_size(batch_index, n_batches, steps)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"BF16 autocast is unsupported on device {device.type}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def ledger_identity(backbone: str, amp: bool, grad_accum_steps: int) -> tuple[str, str, str]:
    if backbone == "resnet18-fpn":
        return "t1-85-resnet-fpn", "85-t1-cnn-fpn", "T1-85 ResNet18-FPN stride 4"
    if backbone == "cnn" and amp and grad_accum_steps == 4:
        return "t1-85-cnn-control", "85-t1-cnn-fpn-control", "T1-85 CNN matched control"
    return "p50-t1-keypoint", "50-t1-keypoint", "T1 keypoint: CNN + heatmap subpixel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=Path, default=Path('checkpoints/task1_keypoint'))
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--n-downsamples', type=int, default=4)
    parser.add_argument('--backbone', choices=('cnn', 'dinov2', 'resnet18-fpn'), default='cnn')
    parser.add_argument('--freeze-backbone', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pretrained-backbone', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--backbone-lr-multiplier', type=float, default=0.1)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help="Enable BF16 autocast (no GradScaler required)")
    parser.add_argument('--upsample-factor', type=int, default=2)
    parser.add_argument('--heatmap-sigma', type=float, default=2.0)
    parser.add_argument('--heatmap-weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-every', type=int, default=1,
                         help="Validar cada N epocas (siempre valida la ultima). La validacion "
                              "sobre MooseFS cuesta comparable a una epoca de train.")
    parser.add_argument('--case-cache', type=Path, default=None,
                         help="Ruta del cache de la lista de casos. Sin esto, cada lanzamiento "
                              "reabre los 61,691 JSON del dataset (~15 min sobre MooseFS).")
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                         help="Techo de clip_grad_norm_. Smoke test (T1-R3, ver ATTACK_LADDER.md) "
                              "midio norma cruda de gradiente hasta 7469 con 1.0 -- casi todos los pasos "
                              "quedan dominados por el clip, no por la senal de la perdida. Subir si la "
                              "convergencia sigue lenta sobre datos reales.")
    parser.add_argument('--num-workers', type=int, default=8,
                         help="Workers de DataLoader para paralelizar I/O sobre MooseFS")
    parser.add_argument('--predictions-out', type=Path, default=None,
                        help="Base JSONL de predicciones val; por default se guarda bajo --out")
    parser.add_argument('--augment', choices=('off', 'light', 'strong'), default='off',
                        help="Intensidad de augmentacion geometrica+fotometrica del fundus de "
                             "TRAIN (ver fido.data.task1_augment). 'off' preserva exactamente "
                             "el comportamiento anterior (sin augmentacion). Nunca se aplica a "
                             "val -- la validacion debe medir generalizacion sobre datos reales, "
                             "no sobre vistas sinteticas.")
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- ledger -----------------------------------------------------------
    # Procedencia de esta corrida (`CONSTITUTION.md` §4: ningún número sin su
    # comando y su commit). Va envuelto porque un fallo del ledger NUNCA puede
    # tumbar un entrenamiento: ocho horas de GPU valen más que un registro.
    try:
        from fido import ledger as _ledger
        _inicio = _ledger.ahora()
    except Exception as _exc:  # noqa: BLE001
        _ledger, _inicio = None, None
        print(f"[ledger] AVISO: no disponible ({_exc}); la corrida no se registrará")

    # load_bscan=False: el modelo de keypoint solo mira el fundus -- cargar los
    # 2 B-scans + 2 mascaras de segmentacion cuesta 356KB de I/O por caso que
    # nunca se usan (ver docstring de Task1Dataset).
    dataset = Task1Dataset(args.root, load_fundus=True, load_bscan=False,
                            cache_path=args.case_cache)
    unique_scenarios = {c['scenario'] for c in dataset.cases}

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

    # Augmentacion SOLO en train (ver fido.data.task1_augment) -- val debe
    # seguir midiendo generalizacion sobre casos reales, no sobre vistas
    # sinteticas, para que keypoint_auc siga siendo comparable con corridas
    # anteriores sin augmentacion (T1-80/T1-85).
    augment_config = get_task1_augment_config(args.augment)
    augmented_train_dataset = AugmentedTask1Dataset(
        train_dataset, augment_config, seed=args.seed
    )

    train_loader = DataLoader(
        augmented_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if args.backbone == 'dinov2':
        model = Task1KeypointDinoV2Model(
            freeze_backbone=args.freeze_backbone,
            upsample_factor=args.upsample_factor,
        ).to(device)
    elif args.backbone == 'resnet18-fpn':
        model = Task1KeypointResNet18FPN(
            pretrained=args.pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
    else:
        model = Task1KeypointModel(
            base_channels=args.base_channels, n_downsamples=args.n_downsamples
        ).to(device)
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, args.lr, args.backbone_lr_multiplier)
    )

    args.out.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_mean_error = float('inf')
    best_epoch = -1
    best_n_val = None

    for epoch in range(1, args.epochs + 1):
        augmented_train_dataset.set_epoch(epoch)
        model.train()
        train_losses = []
        coord_l1s = []
        heatmap_bces = []

        optimizer.zero_grad(set_to_none=True)
        n_train_batches = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            fundus = batch['fundus'].to(device)
            with autocast_context(device, args.amp):
                pred = model(fundus)
                loss, sub_losses = compute_loss(
                    pred,
                    {'keypoint': batch['keypoint'].to(device)},
                    heatmap_sigma=args.heatmap_sigma,
                    heatmap_weight=args.heatmap_weight
                )
                scaled_loss = scale_loss_for_accumulation(
                    loss, batch_index, n_train_batches, args.grad_accum_steps)
            scaled_loss.backward()
            # Obligatorio: un proyecto hermano en esta sesión tuvo un colapso de
            # entrenamiento real (heatmap congelado en un valor exacto, gradiente
            # cero para siempre) por no acotar el gradiente en un modelo de heatmap
            # muy parecido a este.
            if should_optimizer_step(batch_index, n_train_batches, args.grad_accum_steps):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item())
            coord_l1s.append(sub_losses['coord_l1'])
            heatmap_bces.append(sub_losses['heatmap_bce'])

        avg_loss = sum(train_losses) / len(train_losses)
        avg_coord_l1 = sum(coord_l1s) / len(coord_l1s)
        avg_heatmap_bce = sum(heatmap_bces) / len(heatmap_bces)
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} (coord_l1={avg_coord_l1:.4f} heatmap_bce={avg_heatmap_bce:.4f})")

        if epoch % args.val_every != 0 and epoch != args.epochs:
            continue

        predictions_path = (args.predictions_out or args.out / 'val_predictions.jsonl')
        predictions_path = predictions_path.with_name(
            f"{predictions_path.stem}_epoch_{epoch:03d}{predictions_path.suffix}"
        )
        keypoint_auc, mean_error, n_valid = evaluate(
            model, val_loader, device, predictions_path
        )
        print(f"[epoch {epoch}]   val_keypoint_auc={keypoint_auc:.4f}  val_mean_error={mean_error:.2f}px  n={n_valid}")

        # Desempate por error medio: mientras el modelo no baje de 10px, todas
        # las epocas dan auc=0.0000 exacto y `>` deja de discriminar, conservando
        # el PRIMER modelo (el peor). Paso de verdad en T2-R2 el 2026-08-17.
        is_better = not np.isnan(keypoint_auc) and (
            keypoint_auc > best_auc
            or (keypoint_auc == best_auc and mean_error < best_mean_error)
        )
        if is_better:
            best_auc = keypoint_auc
            best_mean_error = mean_error
            best_epoch = epoch
            best_n_val = n_valid
            # Nombre reservado: model_0.pth ya lo usa el checkpoint de distancia
            # (T1-R5, train_task1_unet.py) — evitar que se pisen mientras cada
            # componente se prueba por separado.
            checkpoint = (make_task1_keypoint_payload(
                model, architecture="resnet18-fpn", fold=args.fold, seed=args.seed,
                epoch=epoch, model_kwargs={"fpn_channels": 64,
                                            "heatmap_temperature": 1.0},
            ) if args.backbone == "resnet18-fpn" else model.state_dict())
            torch.save(checkpoint, args.out / 'keypoint_only.pth')
            print(f"  -> nuevo mejor (auc={best_auc:.4f} mean_error={mean_error:.2f}px) en epoch {epoch}, guardado en {args.out / 'keypoint_only.pth'}")

    if best_epoch == -1:
        print("No se pudo evaluar AUC en ninguna época (sin casos válidos en val).")
    else:
        print(f"Mejor AUC: {best_auc:.4f} en época {best_epoch}")

    if _ledger is not None:
        ledger_id, ledger_rung, ledger_title = ledger_identity(
            args.backbone, args.amp, args.grad_accum_steps)
        _ledger.record_run_safe(
            id=_ledger.id_de_corrida(ledger_id, _inicio),
            titulo=ledger_title,
            peldano=ledger_rung,
            script="fido.train.train_task1_keypoint",
            task=1,
            estado="completada" if best_epoch != -1 else "abortada",
            metricas=({"val_keypoint_auc": float(best_auc),
                       "val_mean_error_px": float(best_mean_error)}
                      if best_epoch != -1 else {}),
            epoca=best_epoch if best_epoch != -1 else None,
            epocas_planeadas=args.epochs,
            n_val=best_n_val,
            fold=args.fold,
            n_folds=args.n_folds,
            checkpoint=str(args.out / 'keypoint_only.pth'),
            semilla=args.seed,
            inicio=_inicio,
            notas=(f"backbone={args.backbone}; pretrained={args.pretrained_backbone}; "
                   f"freeze={args.freeze_backbone}; microbatch={args.batch_size}; "
                   f"grad_accum={args.grad_accum_steps}; amp_bf16={args.amp}; "
                   f"augment={args.augment}; "
                   "métricas de la mejor época, no la última."),
        )


if __name__ == '__main__':
    main()
