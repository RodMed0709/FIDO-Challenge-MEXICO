from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fido.data.task2 import Task2Dataset
from fido.data.common import group_kfold_indices
from fido.models.fundus_vessel_seg import FundusVesselUNet, combined_bce_dice_loss, vessel_pixel_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento segmentador de vasos en fundus")
    parser.add_argument("--root", type=Path, required=True, help="Ruta raíz del dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje")
    parser.add_argument("--n-folds", type=int, default=5, help="Número de folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold a usar para validación")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/fundus_vessel_seg"), help="Directorio de salida")
    parser.add_argument("--base-channels", type=int, default=32, help="Canales base del modelo")
    parser.add_argument("--depth", type=int, default=4, help="Profundidad del modelo")
    parser.add_argument("--seed", type=int, default=0, help="Semilla aleatoria")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Workers de DataLoader (antes estaba en 0: single-threaded)")
    parser.add_argument("--enface-cache", type=Path, default=None,
                        help="Cache de en-face precomputado (ver infra/precompute_task2_enface.py)")
    return parser.parse_args()


def main():
    args = parse_args()
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
    args.out.mkdir(parents=True, exist_ok=True)

    # Dataset y splits. include_vessel_enface=False: este entrenamiento es
    # segmentación 2D pura del fundus, no necesita el volumen OCT completo
    # (cargar/decodificar 128 slices por __getitem__ sería trabajo
    # desperdiciado aquí y ralentizaría el entrenamiento sin necesidad).
    # Este entrenamiento es segmentacion 2D del fundus: solo necesita `fundus` y
    # `fundus_vessel_mask`. El comentario anterior afirmaba que
    # `include_vessel_enface=False` evitaba cargar el volumen OCT, pero ese flag
    # solo saltaba `Volume/Segmentation/` -- la proyeccion en-face se calculaba
    # igual, leyendo los 128 PNG del volumen (21.5 MB, el 97% de los bytes por
    # muestra) para tirarlos sin usarlos.
    dataset = Task2Dataset(args.root, include_vessel_enface=False,
                            include_vessel_mask=True,
                            enface_cache_dir=args.enface_cache)
    groups = [c["scenario"] for c in dataset.cases]
    unique_groups = set(groups)

    if len(unique_groups) < args.n_folds:
        logger.warning(
            "Solo %d escenarios únicos < n_folds=%d -> usando TODO el dataset "
            "para train y val (solo smoke test, no válido para elegir hiperparámetros).",
            len(unique_groups), args.n_folds,
        )
        train_idx = list(range(len(dataset)))
        val_idx = list(range(len(dataset)))
    else:
        kfold = group_kfold_indices(groups, n_splits=args.n_folds, seed=args.seed)
        for _ in range(args.fold + 1):
            train_idx, val_idx = next(kfold)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    # Modelo y optimizador
    model = FundusVesselUNet(in_channels=3, base_channels=args.base_channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Medir el desbalance de clases REAL antes de fijar pos_weight (no un
    # número inventado) — mismo patrón que train_task1_unet.py con la cánula.
    # Basta con los primeros hasta 5 batches de train: la fracción de vaso es
    # una propiedad casi constante de la anatomía del fundus, no del batch
    # particular.
    frac_vessel_sum = 0.0
    num_batches = 0
    for batch in train_loader:
        mask = batch["fundus_vessel_mask"]
        frac_vessel_sum += vessel_pixel_fraction(mask)  # ya es float, NO tensor: .item() aquí rompería
        num_batches += 1
        if num_batches >= 5:
            break
    frac_vessel = frac_vessel_sum / num_batches
    pos_weight = (1 - frac_vessel) / max(frac_vessel, 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    logger.info("frac_vessel real (medida sobre %d batches): %.4f, pos_weight: %.4f",
                num_batches, frac_vessel, pos_weight)

    best_val_dice = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # Entrenamiento
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch in train_loader:
            fundus = batch["fundus"].to(device)
            mask = batch["fundus_vessel_mask"].to(device)

            logits = model(fundus)
            loss = combined_bce_dice_loss(logits, mask, pos_weight=pos_weight_tensor, dice_weight=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_total / max(train_batches, 1)
        logger.info("[epoch %d] train_loss=%.6f", epoch, avg_train_loss)

        # Validación
        model.eval()
        val_loss_total = 0.0
        val_dice_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                fundus = batch["fundus"].to(device)
                mask = batch["fundus_vessel_mask"].to(device)

                logits = model(fundus)
                loss = combined_bce_dice_loss(logits, mask, pos_weight=pos_weight_tensor, dice_weight=1.0)
                val_loss_total += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                # Dice por muestra: 2*intersection/(union+eps) sumando sobre H,W,
                # promediado sobre el batch.
                intersection = (preds * mask).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
                dice_per_sample = (2 * intersection) / (union + 1e-6)
                val_dice_total += dice_per_sample.mean().item()
                val_batches += 1

        avg_val_loss = val_loss_total / max(val_batches, 1)
        avg_val_dice = val_dice_total / max(val_batches, 1)
        logger.info("[epoch %d]   val_loss=%.6f val_dice=%.6f", epoch, avg_val_loss, avg_val_dice)

        # Guardar mejor checkpoint
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), args.out / "model.pth")
            logger.info("    -> Mejor checkpoint guardado (val_dice=%.6f)", best_val_dice)

    logger.info("Mejor val_dice: %.6f en época %d", best_val_dice, best_epoch)

    if _ledger is not None:
        _ledger.record_run_safe(
            id=_ledger.id_de_corrida("aux-fundus-vessel-seg", _inicio),
            titulo="Auxiliar: segmentación de vasos en el fundus",
            peldano="auxiliar-fundus-vessel-seg",
            script="fido.train.train_fundus_vessel_seg",
            task=None,
            estado="completada" if best_epoch != -1 else "abortada",
            metricas=({"val_dice": float(best_val_dice)} if best_epoch != -1 else {}),
            epoca=best_epoch if best_epoch != -1 else None,
            epocas_planeadas=args.epochs,
            checkpoint=str(args.out / "model.pth"),
            semilla=args.seed,
            inicio=_inicio,
            notas="Registrado automáticamente. No pertenece a ningún peldaño de experiments/.",
        )


if __name__ == "__main__":
    main()
