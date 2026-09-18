from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fido.data.task2 import Task2Dataset, Task2TransformSubset, load_task2_scales
from fido.data.task2_transforms import Task2GeometricAugment, derive_scale_augmentation
from fido.data.common import group_kfold_indices
from fido.models.task2_baseline import FundusEnfaceHeatmapModel
from fido.models.task2_dinov2 import FundusEnfaceDinoV2Model
from fido.geometry import compose_similarity, corner_error, corner_auc, huber_saturated
from fido.heatmap_decode import gaussian_heatmap_target


def compute_loss(
    pred: dict,
    batch: dict,
    fundus_feat_size: int,
    fundus_size: int = 1024,
) -> tuple[torch.Tensor, dict]:
    pred_matrix = compose_similarity(
        pred["tx"], pred["ty"], pred["cos_theta"], pred["sin_theta"], pred["scale"], reflect=False
    )
    errors = corner_error(pred_matrix, batch["gt_matrix"])
    corner_loss = huber_saturated(errors).mean()

    # El heatmap de correlación marca dónde cae el CENTRO de la plantilla, no
    # la esquina uv=(0,0) que es lo que guarda `target_params[:, :2]` (tx,ty).
    # Están separados por A@(0.5,0.5) — 113.1 px de media sobre los 1214 casos
    # reales. Supervisar el heatmap con (tx,ty) le pedía al modelo apuntar a un
    # punto que la correlación no puede marcar; el GT del heatmap tiene que ser
    # el centro, calculado con la misma matriz GT.
    #
    # El -0.5 en celdas: el kernel de correlación tiene lados PARES, así que la
    # celda de salida `o` corresponde al centro de plantilla `o - 0.5`. El GT
    # tiene que vivir en el mismo sistema de coordenadas que la predicción.
    gt_matrix = batch["gt_matrix"]
    gt_center = gt_matrix[:, :2, :2] @ torch.full(
        (2,), 0.5, device=gt_matrix.device, dtype=gt_matrix.dtype
    ) + gt_matrix[:, :2, 2]

    heat_h, heat_w = pred["heatmap_logits"].shape[-2:]
    stride = fundus_size / fundus_feat_size
    center = gt_center / stride + 0.5
    target_heatmap = gaussian_heatmap_target(
        center,
        heat_h,
        heat_w,
        sigma=1.5,
        device=pred["heatmap_logits"].device,
        dtype=pred["heatmap_logits"].dtype,
    ).unsqueeze(1)
    # pos_weight: el pico gaussiano ocupa ~1.6% del mapa (69 de 4225 celdas con
    # sigma=1.5), así que un BCE con reducción media deja la señal de "dónde
    # está" en el ruido numérico del fondo. Ponderar los positivos la devuelve
    # a una magnitud comparable.
    heatmap_loss = F.binary_cross_entropy_with_logits(
        pred["heatmap_logits"],
        target_heatmap,
        pos_weight=torch.tensor(50.0, device=target_heatmap.device, dtype=target_heatmap.dtype),
        reduction="none",
    )
    heatmap_valid = pred.get("heatmap_valid_mask", torch.ones_like(target_heatmap, dtype=torch.bool))
    heatmap_bce = (heatmap_loss * heatmap_valid).sum() / heatmap_valid.sum().clamp_min(1)

    # La escala se compara en LOG, no en píxeles crudos: el objetivo real ronda
    # 160 px, así que un MSE absoluto sobre la escala valía ~45 al arrancar (el
    # 97% de la pérdida total) y monopolizaba el presupuesto de clip_grad_norm_,
    # dejando a los encoders sin gradiente. En log el error es multiplicativo y
    # queda en el mismo orden que cos/sin.
    param_mse = (
        F.mse_loss(pred["cos_theta"], batch["target_params"][:, 2])
        + F.mse_loss(pred["sin_theta"], batch["target_params"][:, 3])
        + F.mse_loss(torch.log(pred["scale"]), torch.log(batch["target_params"][:, 4]))
    )

    # huber_saturated da gradiente CERO por diseño una vez el error pasa el
    # umbral de saturación (15px, ver geometry.py) — correcto para no perseguir
    # fallos ya catastróficos, pero inútil como única señal de arranque cuando
    # el modelo empieza a cientos de píxeles de error. L1 sin saturar sobre
    # (tx,ty) da gradiente constante en ese régimen, hasta que corner_loss
    # empieza a aportar por sí solo.
    # Normalizado por el tamaño de la imagen: en píxeles crudos esta pérdida
    # producía normas de gradiente de miles contra un `clip_grad_norm_(1.0)`,
    # o sea el clip actuaba como un learning rate absurdamente pequeño y fijo,
    # y lo que parecía "convergencia lenta" era el optimizador congelado.
    coord_l1 = (
        F.l1_loss(pred["tx"] / fundus_size, batch["target_params"][:, 0] / fundus_size)
        + F.l1_loss(pred["ty"] / fundus_size, batch["target_params"][:, 1] / fundus_size)
    )

    # El heatmap sube de 0.1 a 1.0: ahora es la única señal que enseña DÓNDE
    # está la plantilla (antes competía contra un param_mse mal escalado que se
    # llevaba el 97% de la pérdida).
    loss_total = corner_loss + 1.0 * heatmap_bce + 1.0 * param_mse + 10.0 * coord_l1
    return loss_total, {
        "corner": corner_loss.item(),
        "heatmap_bce": heatmap_bce.item(),
        "param_mse": param_mse.item(),
        "coord_l1": coord_l1.item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/task2_baseline"))
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--backbone", choices=("cnn", "dinov2"), default="cnn")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8,
                         help="Workers de DataLoader para paralelizar I/O sobre MooseFS (fs de red del pod, lento en lecturas single-threaded)")
    parser.add_argument("--enface-cache", type=Path, default=None,
                         help="Directorio con las proyecciones en-face precomputadas "
                              "(ver infra/precompute_task2_enface.py). Sin esto, cada caso "
                              "abre los 128 PNG del volumen: ~330x mas I/O.")
    parser.add_argument("--val-every", type=int, default=1,
                         help="Validar cada N epocas (siempre valida la ultima). MooseFS es tan lento en "
                              "lecturas que una sola validacion (248 casos, 128 slices c/u) tardo >1h en la "
                              "practica -- subir esto para no gastar la mayoria del tiempo de entrenamiento "
                              "en I/O de validacion redundante.")
    parser.add_argument("--scale-augment", action=argparse.BooleanOptionalAction, default=False,
                        help="Scale-only exact fundus augmentation; range is derived from train_idx.")
    parser.add_argument("--scale-extrapolation", type=float, default=0.25,
                        help="Fractional expansion of the train-only log-scale quantile span.")
    args = parser.parse_args()
    if args.scale_augment and args.backbone != "cnn":
        parser.error("--scale-augment T2-81 is isolated to the CNN baseline")

    torch.manual_seed(args.seed)

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

    # include_vessel_enface=False: este baseline (T2-R2) no usa el campo
    # enface_vessel (motor de vasos T2-R4 refutado, ver ATTACK_LADDER.md) --
    # cargarlo dobla el I/O por caso (128 PNGs extra de Segmentation) sin
    # ningún beneficio, mismo patrón ya aplicado en train_fundus_vessel_seg.py.
    dataset = Task2Dataset(args.root, include_vessel_enface=False,
                            include_vessel_mask=False,
                            enface_cache_dir=args.enface_cache)
    groups = [c["scenario"] for c in dataset.cases]
    unique_scenarios = set(groups)

    if len(unique_scenarios) < args.n_folds:
        print(
            f"[AVISO] Solo hay {len(unique_scenarios)} escenario(s) - usando train=val completo. "
            "Esto es valido SOLO para un smoke test de overfit, nunca para elegir "
            "hiperparametros ni para una submission real."
        )
        train_idx = list(range(len(dataset)))
        val_idx = list(range(len(dataset)))
    else:
        kfold = group_kfold_indices(groups, n_splits=args.n_folds, seed=args.seed)
        for _ in range(args.fold + 1):
            train_idx, val_idx = next(kfold)

    train_transform = None
    if args.scale_augment:
        scale_config = derive_scale_augmentation(
            load_task2_scales(dataset.cases), train_idx, args.scale_extrapolation,
        )
        train_transform = Task2GeometricAugment(
            scale_config.factor_min, scale_config.factor_max, seed=args.seed,
        )
        config_path = args.out / "scale_augmentation.json"
        args.out.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(asdict(scale_config), indent=2), encoding="utf-8")
        print(f"[scale-augment] factor=[{scale_config.factor_min:.5f}, "
              f"{scale_config.factor_max:.5f}] source_rows={len(scale_config.source_indices)}")

    train_dataset = Task2TransformSubset(dataset, train_idx, train_transform)
    val_dataset = Task2TransformSubset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if args.backbone == "dinov2":
        model = FundusEnfaceDinoV2Model(freeze_backbone=args.freeze_backbone)
    else:
        model = FundusEnfaceHeatmapModel(base_channels=args.base_channels, n_downsamples=4)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    fundus_feat_size = 73 if args.backbone == "dinov2" else 1024 // 2**4
    best_auc = -1.0
    # Desempate obligatorio: mientras el modelo no baje de 10px de error, TODAS
    # las epocas dan auc=0.0000 exacto, y `auc > best_auc` deja de discriminar
    # tras la primera validacion -- se conserva el PRIMER modelo (el peor) y se
    # tiran todos los siguientes. Paso en la corrida real de T2-R2 del 2026-08-17:
    # guardo la epoca 5 (203.29px) y descarto la epoca 25 (98.74px). El error
    # medio si discrimina en ese regimen, asi que desempata.
    best_mean_error = float("inf")
    best_epoch = -1
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_corner = 0.0
        total_heatmap = 0.0
        total_param = 0.0
        total_coord = 0.0
        n_batches = 0

        for batch in train_loader:
            device_batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            if args.backbone == "cnn":
                pred = model(device_batch["fundus"], device_batch["enface"],
                             valid_mask=device_batch.get("valid_mask"))
            else:
                pred = model(device_batch["fundus"], device_batch["enface"])
            loss, sub_losses = compute_loss(pred, device_batch, fundus_feat_size)
            loss.backward()
            # Sin esto, el smoke test de overfit mostró un colapso real: el
            # heatmap de correlación cruzada empieza con logits enormes
            # (heatmap_bce~838 en la época 1), el gradiente inicial revienta,
            # y GroupNorm sobre una salida ya colapsada a una constante la
            # deja fija en 0 para siempre (heatmap_bce = ln(2) exacto de ahí
            # en adelante — BCE con logit=0 en todos lados, sin gradiente).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_corner += sub_losses["corner"]
            total_heatmap += sub_losses["heatmap_bce"]
            total_param += sub_losses["param_mse"]
            total_coord += sub_losses["coord_l1"]
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_corner = total_corner / max(n_batches, 1)
        avg_heatmap = total_heatmap / max(n_batches, 1)
        avg_param = total_param / max(n_batches, 1)
        avg_coord = total_coord / max(n_batches, 1)
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} (corner={avg_corner:.3f} heatmap={avg_heatmap:.4f} param={avg_param:.4f} coord_l1={avg_coord:.2f})")

        is_last_epoch = epoch == args.epochs
        if epoch % args.val_every == 0 or is_last_epoch:
            model.eval()
            all_errors = []
            with torch.no_grad():
                for batch in val_loader:
                    device_batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    if args.backbone == "cnn":
                        pred = model(device_batch["fundus"], device_batch["enface"],
                                     valid_mask=device_batch.get("valid_mask"))
                    else:
                        pred = model(device_batch["fundus"], device_batch["enface"])
                    pred_matrix = compose_similarity(
                        pred["tx"], pred["ty"], pred["cos_theta"], pred["sin_theta"], pred["scale"], reflect=False
                    )
                    errors = corner_error(pred_matrix, device_batch["gt_matrix"])
                    all_errors.extend(errors.cpu().numpy().tolist())

            val_errors = np.array(all_errors)
            auc = corner_auc(val_errors)
            mean_error = val_errors.mean()
            print(f"[epoch {epoch}]   val_corner_auc={auc:.4f}  val_mean_error={mean_error:.2f}px  n={len(val_dataset)}")

            is_better = auc > best_auc or (auc == best_auc and mean_error < best_mean_error)
            if is_better:
                best_auc = auc
                best_mean_error = mean_error
                best_epoch = epoch
                torch.save(model.state_dict(), args.out / "model_1.pth")
                print(
                    f"  -> nuevo mejor (auc={auc:.4f} mean_error={mean_error:.2f}px), "
                    f"guardado en {args.out / 'model_1.pth'}"
                )

    print(f"Mejor AUC: {best_auc:.4f} en epoca {best_epoch}")

    if _ledger is not None:
        _ledger.record_run_safe(
            id=_ledger.id_de_corrida("p40-t2-registracion", _inicio),
            titulo="T2 registración: heatmap por correlación + regresión (theta, s)",
            peldano="40-t2-decodificador-corregido",
            script="fido.train.train_task2_baseline",
            task=2,
            estado="completada" if best_epoch != -1 else "abortada",
            metricas=({"val_corner_auc": float(best_auc),
                       "val_mean_error_px": float(best_mean_error)}
                      if best_epoch != -1 else {}),
            epoca=best_epoch if best_epoch != -1 else None,
            epocas_planeadas=args.epochs,
            n_val=len(val_dataset),
            fold=args.fold,
            n_folds=args.n_folds,
            checkpoint=str(args.out / "model_1.pth"),
            semilla=args.seed,
            inicio=_inicio,
            notas=("Registrado automáticamente al terminar. Métricas de la MEJOR "
                   "época. Si esta corrida pertenece a otro peldaño, corrige el "
                   "campo `peldano` en el YAML."),
        )


if __name__ == "__main__":
    main()
