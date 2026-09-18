#!/usr/bin/env python3
"""T1-93: fraccion real de NaN de `distance_from_segmentation` sobre la
propia prediccion del checkpoint de produccion.

Pregunta que responde: cuando `submissions/r06-fallback-fixed/model_0.pth`
segmenta un B-scan real (no una mascara GT perfecta), ¿en que fraccion de
casos NO encuentra clase instrumento o clase ILM en su propio argmax, y por
lo tanto `distance_from_segmentation` devuelve NaN y el caso cae al
fallback constante?

Distinto de `analysis/measure_t1_distance_rescale_effect.py` (T1-86), que
mide el "techo" usando MASCARAS GT (segmentacion perfecta). Este script mide
lo que pasa con la segmentacion que el modelo realmente produce, que es lo
que ocurre en Codabench.

Datos: `data/Task 1/Scenario_XX.zip` (56 GB total, NO se extraen a disco --
se leen los PNG de B-scan en memoria via `zipfile` + `PIL.Image`) y
`data/_annotations/Task 1/Scenario_XX/*.json` (extraido localmente, barato,
se usa para filtrar canula activa sin tocar el zip).

    python analysis/measure_task1_segmentation_nan_fraction.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Sin esto, cada llamada a `model(...)` intenta usar TODOS los cores logicos
# via el pool de threads interno de PyTorch -- con 3 scripts CPU-bound
# corriendo a la vez (este + measure_t1_distance_rescale_effect.py) eso
# genera contencion/oversubscription severa (miles de CPU-segundos gastados
# en scheduling, casi sin avanzar). 2 threads es suficiente para el batching
# de abajo y dejalibre al resto de procesos.
torch.set_num_threads(2)

from fido.models.unet_bscan_seg import UNet, distance_from_segmentation  # noqa: E402


def mean_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def load_checkpoint(path: Path) -> UNet:
    """`submissions/r06-fallback-fixed/model_0.pth` empaqueta DOS modelos en
    un solo archivo: `raw["keypoint"]` (heatmap CNN) y `raw["distance"]`
    (el UNet de segmentacion que usa `distance_from_segmentation`). Distinto
    del formato `{"state_dict": ...}` de los checkpoints de entrenamiento en
    `checkpoints/` -- este es el bundle real de submission, verificado
    inspeccionando las claves top-level del archivo."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "distance" in raw:
        state = raw["distance"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    state = {k.removeprefix("module."): v for k, v in state.items()}
    base = state["encoders.0.conv1.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("encoders."))
    model = UNet(base_channels=base, depth=depth)
    model.load_state_dict(state)
    model.eval()
    return model


def cannula_active(annotation: dict) -> bool:
    ilm = annotation.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
    return ilm is not None and ilm < 1e6


def collect_active_frame_ids(annotations_root: Path, scenario: str) -> list[str]:
    scen_dir = annotations_root / scenario
    frame_ids = []
    for json_path in sorted(scen_dir.glob("*.json")):
        try:
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cannula_active(annotation):
            frame_ids.append(json_path.stem)
    return frame_ids


def systematic_sample(items: list[str], target: int) -> list[str]:
    if len(items) <= target:
        return items
    stride = len(items) / target
    idx = np.round(np.arange(target) * stride).astype(int)
    idx = np.clip(idx, 0, len(items) - 1)
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def load_bscan_from_zip(zf: zipfile.ZipFile, frame_id: str, stem: str) -> np.ndarray | None:
    name = f"iOCT Microscope/Bscan/{frame_id}/{stem}.png"
    try:
        with zf.open(name) as handle:
            data = handle.read()
    except KeyError:
        return None
    image = Image.open(io.BytesIO(data)).convert("L")
    return np.array(image, dtype=np.uint8)


def classify_nan_reason(seg_logits: torch.Tensor, ilm_class: int = 1, instrument_class: int = 2) -> str:
    """Replica el criterio interno de `distance_from_segmentation` para un
    solo caso (batch=1) y explica por que dio NaN."""
    seg = seg_logits.argmax(dim=1)[0]
    tip_mask = seg == instrument_class
    ilm_mask = seg == ilm_class
    if not tip_mask.any():
        return "sin_instrumento"
    if not ilm_mask.any():
        return "sin_ilm_global"
    tip_rows, tip_cols = torch.nonzero(tip_mask, as_tuple=True)
    max_row_idx = tip_rows.argmax()
    tip_col = int(tip_cols[max_row_idx])
    col_min = max(0, tip_col - 5)
    col_max = min(seg.shape[1] - 1, tip_col + 5)
    ilm_in_window = ilm_mask[:, col_min:col_max + 1]
    if not ilm_in_window.any():
        return "sin_ilm_en_ventana"
    return "medido"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "Task 1")
    parser.add_argument("--annotations-root", type=Path,
                        default=ROOT / "data" / "_annotations" / "Task 1")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "submissions" / "r06-fallback-fixed" / "model_0.pth")
    parser.add_argument("--target-per-scenario", type=int, default=550)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments" / "93-t1-dfl" / "NAN_FRACTION_MEASUREMENT.md")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = load_checkpoint(args.checkpoint)

    scenarios = sorted(p.name for p in args.annotations_root.iterdir() if p.is_dir())
    started = time.time()

    per_scenario: dict[str, dict] = {}
    reason_counts: dict[str, int] = {}
    total_sampled = total_with_oct = total_measured = 0

    CASES_PER_BATCH = 4  # hasta 8 slices de 512x512 por forward -- UNet sin
                         # downsampling inicial es caro en memoria a full
                         # resolucion (un batch de 32 imagenes ya pasa de
                         # varios GB de activaciones); 4 casos amortiza el
                         # overhead de threading sin reventar RAM/CPU.

    for scenario in scenarios:
        active_ids = collect_active_frame_ids(args.annotations_root, scenario)
        sample_ids = systematic_sample(active_ids, args.target_per_scenario)
        zip_path = args.data_root / f"{scenario}.zip"
        n_measured = n_no_oct = n_sampled = 0
        with zipfile.ZipFile(zip_path) as zf:
            for batch_start in range(0, len(sample_ids), CASES_PER_BATCH):
                batch_ids = sample_ids[batch_start:batch_start + CASES_PER_BATCH]
                case_slices: list[list[torch.Tensor]] = []
                for frame_id in batch_ids:
                    slice0 = load_bscan_from_zip(zf, frame_id, "00")
                    slice1 = load_bscan_from_zip(zf, frame_id, "01")
                    if slice0 is None and slice1 is None:
                        n_no_oct += 1
                        case_slices.append([])
                        continue
                    n_sampled += 1
                    tensors = [torch.from_numpy(arr.astype(np.float32) / 255.0)
                              for arr in (slice0, slice1) if arr is not None]
                    case_slices.append(tensors)

                flat = [t for tensors in case_slices for t in tensors]
                if not flat:
                    continue
                stacked = torch.stack(flat).unsqueeze(1)  # (n_slices_total, 1, 512, 512)
                with torch.inference_mode():
                    logits = model(stacked)

                offset = 0
                for tensors in case_slices:
                    if not tensors:
                        continue
                    n_slices = len(tensors)
                    gaps, reasons = [], []
                    for s in range(n_slices):
                        single = logits[offset + s:offset + s + 1]
                        gap = distance_from_segmentation(single).item()
                        gaps.append(gap)
                        reasons.append(classify_nan_reason(single))
                    offset += n_slices
                    case_gap = mean_finite(gaps)
                    if np.isfinite(case_gap):
                        n_measured += 1
                        reason_counts["medido"] = reason_counts.get("medido", 0) + 1
                    else:
                        # Motivo del caso = el primer motivo no-"medido" (ambas
                        # slices fallaron; si una hubiera dado finito, case_gap
                        # seria finito).
                        reason = next((r for r in reasons if r != "medido"), "desconocido")
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        per_scenario[scenario] = {
            "n_active_total": len(active_ids),
            "n_sampled": n_sampled,
            "n_no_oct_in_sample": n_no_oct,
            "n_measured": n_measured,
            "n_nan": n_sampled - n_measured,
        }
        total_sampled += n_sampled
        total_with_oct += n_sampled
        total_measured += n_measured
        elapsed = time.time() - started
        print(f"[{elapsed:7.1f}s] {scenario}: activos={len(active_ids):,} "
              f"muestreados={n_sampled:,} medidos={n_measured:,} "
              f"NaN={n_sampled - n_measured:,}", flush=True)

    total_nan = total_sampled - total_measured
    nan_fraction = total_nan / total_sampled if total_sampled else float("nan")

    lines = [
        "# T1-93 -- fraccion real de NaN de `distance_from_segmentation`",
        "",
        f"**Checkpoint:** `{args.checkpoint}`  ",
        f"**Muestra:** sistematica, objetivo {args.target_per_scenario} casos/escenario, "
        f"canula activa, con al menos un B-scan presente en el zip.  ",
        f"**Total muestreado:** {total_sampled:,}  ",
        f"**Medido (finito):** {total_measured:,}  ",
        f"**NaN (fallback):** {total_nan:,}  ",
        f"**Fraccion NaN:** {nan_fraction:.4f} ({nan_fraction:.1%})  ",
        f"**Tiempo total:** {time.time() - started:.1f}s  ",
        "",
        "## Desglose por motivo (sobre los NaN)",
        "",
        "| motivo | n | fraccion del total muestreado |",
        "|---|---:|---:|",
    ]
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        if reason == "medido":
            continue
        lines.append(f"| {reason} | {count:,} | {count / total_sampled:.4f} |")
    lines += [
        "",
        "## Por escenario",
        "",
        "| escenario | n activos (total) | n muestreado | n medido | n NaN | fraccion NaN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, row in per_scenario.items():
        frac = row["n_nan"] / row["n_sampled"] if row["n_sampled"] else float("nan")
        lines.append(f"| {scenario} | {row['n_active_total']:,} | {row['n_sampled']:,} | "
                     f"{row['n_measured']:,} | {row['n_nan']:,} | {frac:.4f} |")
    lines += [
        "",
        "## Reproduccion",
        "",
        "```bash",
        "python analysis/measure_task1_segmentation_nan_fraction.py",
        "```",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
