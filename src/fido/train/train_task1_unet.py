from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fido.data.task1 import Task1Dataset, task1_split
from fido.eval_task1 import evaluate_distances
from fido.eval_task1 import write_evaluation_outputs
from fido.models.unet_bscan_seg import UNet, distance_from_segmentation

@dataclass(frozen=True)
class DistanceCalibration:
    scale: float
    offset: float
    fallback: float
    fit_case_ids: tuple[str, ...]


def calibration_to_dict(calibration: DistanceCalibration) -> dict:
    return {
        "scale": calibration.scale,
        "offset": calibration.offset,
        "fallback": calibration.fallback,
        "fit_case_ids": list(calibration.fit_case_ids),
    }


def calibration_from_dict(payload: dict) -> DistanceCalibration:
    return DistanceCalibration(
        scale=float(payload["scale"]), offset=float(payload["offset"]),
        fallback=float(payload["fallback"]),
        fit_case_ids=tuple(str(value) for value in payload["fit_case_ids"]),
    )


def save_distance_calibration(path: Path, calibration: DistanceCalibration) -> None:
    path.write_text(json.dumps(calibration_to_dict(calibration), indent=2) + "\n", encoding="utf-8")


def load_distance_calibration(path: Path) -> DistanceCalibration:
    return calibration_from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_distance_calibration_samples(gaps, distances, case_ids) -> DistanceCalibration:
    gaps_array = np.asarray(gaps, dtype=np.float64)
    distance_array = np.asarray(distances, dtype=np.float64)
    ids = tuple(str(case_id) for case_id in case_ids)
    if gaps_array.shape != distance_array.shape or gaps_array.ndim != 1:
        raise ValueError("gaps and distances must have matching one-dimensional shapes")
    if len(ids) != len(gaps_array) or len(set(ids)) != len(ids):
        raise ValueError("calibration case IDs must be unique and match sample count")
    valid = np.isfinite(gaps_array) & np.isfinite(distance_array)
    if valid.sum() < 2 or np.ptp(gaps_array[valid]) == 0:
        raise ValueError("at least two distinct finite train gaps are required")
    scale, offset = np.polyfit(gaps_array[valid], distance_array[valid], deg=1)
    finite_gt = distance_array[np.isfinite(distance_array)]
    if finite_gt.size == 0:
        raise ValueError("at least one finite train distance is required for fallback")
    return DistanceCalibration(float(scale), float(offset),
                               float(np.median(finite_gt)), ids)


def fit_distance_calibration(dataloader: DataLoader, device: torch.device) -> DistanceCalibration:
    gaps, distances, case_ids = [], [], []
    for batch in dataloader:
        seg = batch["bscan_seg"].to(device)
        batch_distances = batch["distance"].cpu().numpy()
        B, n_slices, _, _ = seg.shape
        for i in range(B):
            sample_gaps = []
            for s in range(n_slices):
                if not (seg[i, s] >= 0).any():
                    continue
                labels = seg[i, s].clamp(min=0)
                logits = F.one_hot(labels, num_classes=3).permute(2, 0, 1).float().unsqueeze(0)
                gap = distance_from_segmentation(logits)[0]
                if torch.isfinite(gap):
                    sample_gaps.append(float(gap))
            gaps.append(float(np.mean(sample_gaps)) if sample_gaps else float("nan"))
            distances.append(float(batch_distances[i]))
            case_ids.append(f"{batch['scenario'][i]}/{batch['frame_id'][i]}")
    return fit_distance_calibration_samples(gaps, distances, case_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/task1_unet"))
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8,
                         help="Workers de DataLoader para paralelizar I/O sobre MooseFS (fs de red del pod, lento en lecturas single-threaded)")
    parser.add_argument("--val-every", type=int, default=1,
                         help="Validar cada N epocas (siempre valida la ultima)")
    parser.add_argument("--case-cache", type=Path, default=None,
                         help="Ruta del cache de la lista de casos. Sin esto, cada lanzamiento "
                              "reabre los 61,691 JSON del dataset (~15 min sobre MooseFS).")
    parser.add_argument("--class-weights", type=str, default=None,
                         help="3 floats separados por coma (fondo,ilm,canula) para saltar por completo "
                              "el escaneo de class_counts -- util cuando MooseFS esta lento. Valor real "
                              "medido en una corrida anterior sobre datos completos: "
                              "'0.0341661,0.4904578,2.4753761'")
    parser.add_argument("--allow-transductive-class-weights", action="store_true",
                        help="Marca como transductivo el uso de pesos historicos globales")
    parser.add_argument("--predictions-out", type=Path, default=None,
                        help="Base JSONL de predicciones val; por default se guarda bajo --out")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    calibration: DistanceCalibration,
    predictions_out: Path | None = None,
) -> tuple[float, float, float, int]:
    """Evalúa segmentación pixel-accuracy y distancia MAE/AUC en val."""
    model.eval()
    total_correct = 0
    total_pixels = 0
    all_predictions: list[float] = []
    all_ground_truth: list[float] = []
    all_scenarios: list[str] = []
    all_measured: list[bool] = []
    all_case_ids: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            bscan = batch["bscan"].to(device)
            bscan_seg = batch["bscan_seg"].to(device)
            has_oct = batch["has_oct"].cpu().numpy()
            distances = batch["distance"].cpu().numpy()

            B, n_slices, H, W = bscan.shape
            bscan_flat = bscan.reshape(B * n_slices, 1, H, W)
            seg_flat = bscan_seg.reshape(B * n_slices, H, W)
            logits_flat = model(bscan_flat)
            logits = logits_flat.reshape(B, n_slices, 3, H, W)

            # Pixel accuracy ignoring -1
            valid_mask = seg_flat != -1
            if valid_mask.any():
                preds = logits_flat.argmax(dim=1)
                correct = (preds == seg_flat) & valid_mask
                total_correct += correct.sum().item()
                total_pixels += valid_mask.sum().item()

            # Distancia all-case: los casos sin medicion se conservan.
            for i in range(B):
                gaps = []
                if has_oct[i]:
                    for s in range(n_slices):
                        # El slice mantiene dim0 (batch=1); no dejar un tensor 5D.
                        gap = distance_from_segmentation(logits[i:i + 1, s])
                        if not torch.isnan(gap).item():
                            gaps.append(gap.item())
                measured = bool(gaps)
                pred_distance = (calibration.scale * float(np.mean(gaps)) + calibration.offset
                                 if measured else calibration.fallback)
                all_predictions.append(pred_distance)
                all_ground_truth.append(float(distances[i]))
                all_scenarios.append(batch["scenario"][i])
                all_measured.append(measured)
                all_case_ids.append(f"{batch['scenario'][i]}/{batch['frame_id'][i]}")

    pixel_acc = total_correct / total_pixels if total_pixels > 0 else float("nan")
    if len(all_predictions) > 0:
        metrics = evaluate_distances(
            all_predictions, all_ground_truth, all_scenarios, all_measured,
            fallback=calibration.fallback,
        )
        mean_error = metrics["mean_error"]
        distance_auc = metrics["auc"]
        if predictions_out is not None:
            write_evaluation_outputs(predictions_out, metrics, all_case_ids)
    else:
        mean_error = float("nan")
        distance_auc = float("nan")
    return pixel_acc, mean_error, distance_auc, len(all_predictions)


def main() -> None:
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

    # load_fundus=False: este entrenamiento segmenta B-scans, nunca toca el
    # fundus -- cargarlo cuesta 699KB de I/O y un tensor float32 de 12MB por
    # caso (100MB por batch de 8 viajando por IPC) para ser descartado.
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

    model = UNet(in_channels=1, num_classes=3, base_channels=args.base_channels, depth=args.depth)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Desbalance de clases real y medido: en un smoke test de overfit sobre
    # el Mock Test, fondo=99.5%, Ilm=0.46%, cánula=0.008% de los píxeles —
    # con CrossEntropyLoss SIN pesos, el modelo converge a predecir fondo en
    # el 100% de los píxeles incluso memorizando las mismas 5 muestras sobre
    # las que entrena. Los pesos se calculan de los datos reales de esta
    # corrida (frecuencia inversa), no son un número inventado.
    # Muestra de hasta 200 batches (no el train set completo, ~49k casos) --
    # la fraccion de clase es una propiedad casi constante de la anatomia,
    # no de la muestra particular (mismo razonamiento que train_fundus_vessel_seg.py
    # con 5 batches); recorrer TODO el dataset aqui costaba una hora+ de I/O
    # sobre MooseFS antes de que empezara siquiera la epoca 1.
    if args.class_weights is not None and not args.allow_transductive_class_weights:
        raise ValueError(
            "--class-weights is forbidden for clean T1-80 because it may encode "
            "full-dataset statistics; omit it or pass --allow-transductive-class-weights"
        )
    if args.class_weights is not None:
        class_weights = torch.tensor(
            [float(x) for x in args.class_weights.split(",")], dtype=torch.float32
        ).to(device)
        print(f"Pesos de clase pasados por CLI (sin escanear el dataset): {class_weights.tolist()}")
    else:
        class_counts = torch.zeros(3, dtype=torch.float64)
        for i, batch in enumerate(train_loader):
            if i >= 200:
                break
            seg = batch["bscan_seg"]
            valid = seg[seg != -1]
            if valid.numel() > 0:
                class_counts += torch.bincount(valid, minlength=3).double()
        class_weights = 1.0 / torch.sqrt(class_counts.clamp(min=1.0))
        class_weights = (class_weights / class_weights.sum() * 3).float().to(device)
        print(f"Conteo de píxeles por clase (muestra de hasta 200 batches, train): {class_counts.tolist()}")
        print(f"Pesos de clase usados en la pérdida: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    calibration = fit_distance_calibration(train_loader, device)
    val_ids = {f"{dataset.cases[int(i)]['scenario']}/{dataset.cases[int(i)]['frame_id']}"
               for i in val_idx}
    if set(calibration.fit_case_ids) & val_ids:
        raise RuntimeError("distance calibration leaked validation case IDs")
    print(
        f"Calibracion train-only: scale={calibration.scale:.6f} "
        f"offset={calibration.offset:.6f} fallback={calibration.fallback:.3f} "
        f"n_fit={len(calibration.fit_case_ids)}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_distance_calibration(args.out / "distance_calibration.json", calibration)

    best_auc = -1.0
    # Ver comentario equivalente en train_task2_baseline.py: si el AUC se satura
    # (todas las epocas dan el mismo valor exacto), `auc > best_auc` conserva el
    # PRIMER modelo y descarta todos los siguientes. El error medio desempata.
    best_mean_error = float("inf")
    best_epoch = -1
    best_n_val = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            bscan = batch["bscan"].to(device)
            bscan_seg = batch["bscan_seg"].to(device)

            B, n_slices, H, W = bscan.shape
            bscan_flat = bscan.reshape(B * n_slices, 1, H, W)
            seg_flat = bscan_seg.reshape(B * n_slices, H, W)

            # Si ningún píxel del batch es válido (todas las muestras tienen
            # has_oct=False), CrossEntropyLoss(ignore_index=-1) divide entre
            # cero -> NaN. Y un solo paso con gradiente NaN corrompe los
            # pesos PARA SIEMPRE (clip_grad_norm_ no salva un NaN, sigue
            # siendo NaN después de escalar). Saltar el batch entero es más
            # seguro que intentar filtrar dentro del batch.
            if not (seg_flat != -1).any():
                continue

            optimizer.zero_grad()
            logits_flat = model(bscan_flat)
            loss = criterion(logits_flat, seg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        if not train_losses:
            print(f"[epoch {epoch}] sin batches con píxeles válidos, se saltó la época entera")
            continue

        avg_loss = float(np.mean(train_losses))
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f}")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            predictions_path = (args.predictions_out or args.out / "val_predictions.jsonl")
            predictions_path = predictions_path.with_name(
                f"{predictions_path.stem}_epoch_{epoch:03d}{predictions_path.suffix}"
            )
            val_pixel_acc, val_mean_error, val_auc, n_valid = evaluate(
                model, val_loader, device, calibration, predictions_path
            )
            if n_valid == 0:
                print(
                    f"[epoch {epoch}]   val_pixel_acc={val_pixel_acc:.4f}  val_distance_mae={val_mean_error:.2f}px  "
                    f"val_distance_auc=NaN  n={n_valid} (sin casos OCT válidos)"
                )
            else:
                print(
                    f"[epoch {epoch}]   val_pixel_acc={val_pixel_acc:.4f}  val_distance_mae={val_mean_error:.2f}px  "
                    f"val_distance_auc={val_auc:.4f}  n={n_valid}"
                )

            is_better = not np.isnan(val_auc) and (
                val_auc > best_auc
                or (val_auc == best_auc and val_mean_error < best_mean_error)
            )
            if is_better:
                best_auc = val_auc
                best_mean_error = val_mean_error
                best_epoch = epoch
                best_n_val = n_valid
                torch.save(model.state_dict(), args.out / "model_0.pth")
                print(
                    f"  -> nuevo mejor (auc={best_auc:.4f} mae={val_mean_error:.2f}px) "
                    f"en epoch {epoch}, checkpoint guardado"
                )

    if best_epoch == -1:
        print("No se pudo evaluar AUC en ninguna época (sin casos OCT válidos en val).")
    else:
        print(f"Mejor AUC: {best_auc:.4f} en época {best_epoch}")

    if _ledger is not None:
        _ledger.record_run_safe(
            id=_ledger.id_de_corrida("p10-t1-distancia", _inicio),
            titulo="T1 distancia: UNet cánula+ILM + medición geométrica",
            peldano="10-t1-distancia-unet",
            script="fido.train.train_task1_unet",
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
            notas=("Registrado automáticamente al terminar. Las métricas son las de "
                   "la MEJOR época, no las de la última."),
        )


if __name__ == "__main__":
    main()
