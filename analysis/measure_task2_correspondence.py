#!/usr/bin/env python3
"""Mide la correspondencia densa del baseline actual de Task 2, sin entrenar."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fido.data.common import group_kfold_indices  # noqa: E402
from fido.data.task2 import Task2Dataset, find_task2_cases  # noqa: E402
from fido.eval_task2 import require_task2_training_root, write_case_jsonl  # noqa: E402
from fido.models.task2_baseline import SCALE_REF, FundusEnfaceHeatmapModel  # noqa: E402
from fido.losses.task2_contrastive import sample_positive_pairs  # noqa: E402


def common_descriptor_ranks(fundus_desc: torch.Tensor, oct_desc: torch.Tensor,
                            gt: torch.Tensor, valid_mask: torch.Tensor, k: int,
                            fundus_image_size: tuple[int, int]) -> list[int]:
    """M1 ranks for T2-82 descriptors, using the shared audited sampler."""
    pairs = sample_positive_pairs(fundus_desc, oct_desc, gt, valid_mask, k,
                                  fundus_image_size=fundus_image_size)
    ranks = []
    for index in range(len(pairs.positive)):
        candidates = fundus_desc[pairs.batch_index[index]].flatten(1).T
        scores = candidates @ pairs.oct[index]
        ranks.append(int((scores > pairs.positive[index]).sum().item() + 1))
    return ranks


def build_oct_derangement(indices: list[int] | np.ndarray) -> np.ndarray:
    """Rotación determinista: mismos miembros/orden target y ningún auto-par."""
    values = np.asarray(indices)
    if len(values) < 2:
        raise ValueError("OCT derangement requires at least two cases")
    return np.roll(values, 1)


def resolve_root(requested: Path | None) -> Path:
    """El Mock Test local tiene un nivel ``Task 2`` adicional."""
    candidates = [requested] if requested is not None else [Path("data/Task 2")]
    for candidate in candidates:
        if candidate is None:
            continue
        if find_task2_cases(candidate):
            return candidate
        nested = candidate / "Task 2"
        if find_task2_cases(nested):
            return nested
    raise RuntimeError("No hay casos de entrenamiento utilizables en data/Task 2")


def correlation_map(fundus_feat: torch.Tensor, enface_feat: torch.Tensor) -> torch.Tensor:
    """Replica literalmente la correlación y normalización del modelo."""
    batch_size, channels, hf, wf = fundus_feat.shape
    _, _, ht, wt = enface_feat.shape
    corr = F.conv2d(
        fundus_feat.reshape(1, batch_size * channels, hf, wf),
        enface_feat,
        groups=batch_size,
        padding=(ht // 2, wt // 2),
    )
    out_h = hf + 2 * (ht // 2) - ht + 1
    out_w = wf + 2 * (wt // 2) - wt + 1
    logits = corr.reshape(batch_size, 1, out_h, out_w) / math.sqrt(channels)
    mean = logits.mean(dim=(2, 3), keepdim=True)
    std = logits.std(dim=(2, 3), keepdim=True)
    return (logits - mean) / (std + 1e-6)


def sample_valid_points(gt: torch.Tensor, k: int, rng: np.random.Generator,
                        fundus_h: int, fundus_w: int) -> np.ndarray:
    """Muestrea uniformemente en el cuadrado canónico y rechaza puntos fuera del fundus."""
    accepted: list[np.ndarray] = []
    while sum(len(chunk) for chunk in accepted) < k:
        uv = rng.random((max(k * 2, 128), 2))
        uv_h = np.column_stack([uv, np.ones(len(uv))])
        xy = uv_h @ gt.cpu().numpy().T
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < fundus_w) &
                 (xy[:, 1] >= 0) & (xy[:, 1] < fundus_h))
        accepted.append(uv[valid])
        if sum(len(chunk) for chunk in accepted) == 0 and len(accepted) >= 100:
            raise RuntimeError("La transformación GT no proyecta área válida dentro del fundus")
    return np.concatenate(accepted, axis=0)[:k]


def dense_retrieval(fundus_feat: torch.Tensor, enface_feat: torch.Tensor,
                    gt: torch.Tensor, k: int, rng: np.random.Generator,
                    fundus_h: int, fundus_w: int) -> tuple[list[int], list[int], int]:
    """Devuelve ranks 1-based para positivo vecino y positivo bilineal."""
    _, _, hf, wf = fundus_feat.shape
    uv = sample_valid_points(gt, k, rng, fundus_h, fundus_w)
    uv_t = torch.as_tensor(uv, dtype=enface_feat.dtype, device=enface_feat.device)
    enface_grid = (uv_t * 2.0 - 1.0).view(1, k, 1, 2)
    queries = F.grid_sample(enface_feat, enface_grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[0, :, :, 0].T
    queries = F.normalize(queries, dim=-1)

    ones = torch.ones((k, 1), dtype=gt.dtype, device=gt.device)
    uv_h = torch.cat([uv_t.to(gt.dtype), ones], dim=1)
    xy = uv_h @ gt.T
    x, y = xy[:, 0], xy[:, 1]

    candidates = F.normalize(fundus_feat[0].flatten(1).T, dim=-1)
    scores = queries @ candidates.T
    ix = torch.clamp(torch.floor(x / fundus_w * wf).long(), 0, wf - 1)
    iy = torch.clamp(torch.floor(y / fundus_h * hf).long(), 0, hf - 1)
    nearest_scores = scores[torch.arange(k, device=scores.device), iy * wf + ix]
    nearest_ranks = 1 + (scores > nearest_scores[:, None]).sum(dim=1)

    fundus_grid = torch.stack([x / fundus_w * 2.0 - 1.0,
                               y / fundus_h * 2.0 - 1.0], dim=1).view(1, k, 1, 2)
    positives = F.grid_sample(fundus_feat, fundus_grid, mode="bilinear",
                              padding_mode="border", align_corners=False)[0, :, :, 0].T
    bilinear_scores = (queries * F.normalize(positives, dim=-1)).sum(dim=1)
    bilinear_ranks = 1 + (scores > bilinear_scores[:, None]).sum(dim=1)
    return nearest_ranks.cpu().tolist(), bilinear_ranks.cpu().tolist(), hf * wf


def fmt(value: float, digits: int = 3) -> str:
    return "N/D" if not np.isfinite(value) else f"{value:.{digits}f}"


def is_mock_test(root: Path) -> bool:
    return any(part.lower() == "mock test" for part in root.parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints_from_pod/latest/task2_fixed.pth"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/80-t2-diagnostics/correspondence.md"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--enface-cache", type=Path, default=None)
    parser.add_argument("--oct-shuffle", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = resolve_root(args.root)
    require_task2_training_root(root)
    if is_mock_test(root):
        # CONSTITUTION.md §I.3 prohíbe inspeccionar etiquetas del Mock Test más
        # allá del score agregado. Se cuenta disponibilidad por rutas, pero no
        # se instancia el dataset: hacerlo cargaría la matriz GT por caso.
        available = len(find_task2_cases(root))
        command = "python " + " ".join(shlex.quote(arg) for arg in sys.argv)
        report = f"""# Baseline de correspondencia Task 2 — T2-R10

## Corrida

- Comando exacto: `{command}`
- Checkpoint solicitado: `{args.checkpoint}`
- Dispositivo: `{args.device}`
- Raíz examinada: `{root}`
- Casos disponibles por estructura: **{available}**
- Casos medidos: **0**

## Resultado

No fue posible medir el baseline local. `data/Task 2` no contiene casos extraídos (solo ZIPs),
y los únicos {available} casos utilizables están en Mock Test. `CONSTITUTION.md` §I.3 prohíbe
inspeccionar sus etiquetas más allá del score agregado. El script se detuvo antes de instanciar
`Task2Dataset` o cargar matrices GT.

| Métrica | Definición | Resultado |
|---|---|---:|
| M1 | Recall top-1% y percentil mediano del rank positivo | **No medido** |
| M2 | Recall del centro ≤10 px y distancia mediana, con θ/s GT | **No medido** |
| M3 | Masa softmax del pico y correlación con corrección M2 | **No medido** |

## ¿Reproduce 28.6%?

**NO SE PUEDE VERIFICAR LOCALMENTE.** Se conserva **28.6%** como cifra citada, no como baseline
reproducido por esta corrida.

## Gate propuesto para T2-R10

Hasta ejecutar este mismo script sobre los casos extraídos de `data/Task 2`, los únicos umbrales
numéricos defendibles son los pre-registrados en la literatura:

- **M1:** recall top-1% ≥ **50.0%**.
- **M2:** recall ≤10 px **claramente superior a 28.6%**; propuesta operativa: ≥ **50.0%**.
- **M3:** no fijar umbral derivado sin baseline; como piso provisional de señal útil, correlación
  confianza/corrección ≥ **0.20**, estimada sobre un fold que contenga casos correctos e incorrectos.

Estos umbrales son provisionales, no derivados de una medición local. Deben recalcularse y aprobarse
cuando estén disponibles los datos de entrenamiento extraídos.
"""
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"data_root={root} available_cases={available} measured_cases=0")
        print("M1=NO_MEDIDO M2=NO_MEDIDO M3=NO_MEDIDO")
        print("motivo=CONSTITUTION.md §I.3 prohíbe usar etiquetas del Mock Test por caso")
        print(f"report={args.output}")
        return

    dataset = Task2Dataset(root, include_vessel_enface=False, include_vessel_mask=False,
                           enface_cache_dir=args.enface_cache)
    groups = [case["scenario"] for case in dataset.cases]
    unique_groups = len(set(groups))
    used_fallback = unique_groups < args.n_folds
    if used_fallback:
        val_idx = list(range(len(dataset)))
    else:
        splits = group_kfold_indices(groups, n_splits=args.n_folds, seed=args.seed)
        for _ in range(args.fold + 1):
            _, val_idx = next(splits)

    loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False, num_workers=0)
    model = FundusEnfaceHeatmapModel(base_channels=32, n_downsamples=4)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device).eval()

    nearest_ranks: list[int] = []
    bilinear_ranks: list[int] = []
    center_distances: list[float] = []
    peak_masses: list[float] = []
    correct: list[bool] = []
    per_case: list[tuple[str, float, float, bool]] = []
    n_locations = 0

    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            fundus = batch["fundus"].to(args.device)
            enface = batch["enface"].to(args.device)
            gt = batch["gt_matrix"].to(args.device)
            resized = F.interpolate(enface, size=(int(SCALE_REF), int(SCALE_REF)),
                                    mode="bilinear", align_corners=False)
            fundus_feat = model.fundus_encoder(fundus)
            enface_feat = model.enface_encoder(resized)

            ranks_n, ranks_b, n_locations = dense_retrieval(
                fundus_feat, enface_feat, gt[0], args.k, rng,
                fundus.shape[-2], fundus.shape[-1],
            )
            nearest_ranks.extend(ranks_n)
            bilinear_ranks.extend(ranks_b)

            logits = correlation_map(fundus_feat, enface_feat)
            flat = logits.flatten(1)
            peak_index = flat.argmax(dim=1)
            peak_y = torch.div(peak_index, logits.shape[-1], rounding_mode="floor")
            peak_x = peak_index % logits.shape[-1]
            stride_x = fundus.shape[-1] / fundus_feat.shape[-1]
            stride_y = fundus.shape[-2] / fundus_feat.shape[-2]
            peak_center = torch.stack([(peak_x.float() - 0.5) * stride_x,
                                       (peak_y.float() - 0.5) * stride_y], dim=1)
            half = torch.tensor([0.5, 0.5], dtype=gt.dtype, device=gt.device)
            true_center = gt[:, :2, :2] @ half + gt[:, :2, 2]
            distance = torch.linalg.vector_norm(peak_center - true_center, dim=1).item()
            mass = torch.softmax(flat, dim=1).max(dim=1).values.item()
            is_correct = distance <= 10.0
            center_distances.append(distance)
            peak_masses.append(mass)
            correct.append(is_correct)
            case_id = f"{batch['scenario'][0]}/{batch['frame_id'][0]}"
            per_case.append((case_id, distance, mass, is_correct))
            print(f"[{index}/{len(val_idx)}] {case_id}: M2_dist={distance:.2f}px "
                  f"peak_mass={mass:.6f} correct={is_correct}", flush=True)

    nearest = np.asarray(nearest_ranks)
    bilinear = np.asarray(bilinear_ranks)
    distances = np.asarray(center_distances)
    masses = np.asarray(peak_masses)
    correct_array = np.asarray(correct, dtype=float)
    top_count = math.ceil(0.01 * n_locations)
    m1_recall = float(np.mean(nearest <= top_count))
    m1_percentile = float(np.median(nearest / n_locations * 100.0))
    m1_bilinear_recall = float(np.mean(bilinear <= top_count))
    m1_bilinear_percentile = float(np.median(bilinear / n_locations * 100.0))
    m2_recall = float(np.mean(distances <= 10.0))
    m2_median = float(np.median(distances))
    if len(masses) >= 2 and np.std(masses) > 0 and np.std(correct_array) > 0:
        m3_corr = float(np.corrcoef(masses, correct_array)[0, 1])
    else:
        m3_corr = float("nan")

    command = "python " + " ".join(shlex.quote(arg) for arg in sys.argv)
    split_note = (f"Respaldo del entrenamiento: train=val completo porque solo hay {unique_groups} "
                  f"grupo(s), menos que `n_folds={args.n_folds}`."
                  if used_fallback else f"GroupKFold exacto: fold {args.fold}/{args.n_folds}.")
    reproduces = abs(m2_recall - 0.286) <= 0.005
    reproduction = ("Sí" if reproduces else
                    f"**NO. El baseline medido es {m2_recall * 100:.1f}% frente al 28.6% citado.**")
    gate_m1 = max(0.50, m1_recall + 0.10)
    gate_m2 = max(0.50, m2_recall + 0.10)
    gate_m3 = 0.20

    rows = "\n".join(
        f"| `{case}` | {distance:.2f} | {mass:.6f} | {'sí' if ok else 'no'} |"
        for case, distance, mass, ok in per_case
    )
    report = f"""# Baseline de correspondencia Task 2 — T2-R10

## Corrida

- Comando exacto: `{command}`
- Checkpoint: `{args.checkpoint}`
- Dispositivo: `{args.device}`
- Raíz efectiva: `{root}`
- Casos medidos: **{len(val_idx)}** de {unique_groups} escenario(s)
- Split: {split_note}
- Semilla: {args.seed}; K={args.k} puntos por caso

## Resultados

| Métrica | Definición | Resultado |
|---|---|---:|
| M1, positivo vecino | Positivo entre el top-1% ({top_count}/{n_locations}) de todas las celdas fundus; rank mediano como percentil | recall **{m1_recall * 100:.2f}%**; percentil **{m1_percentile:.3f}%** |
| M1, positivo bilineal | Misma consulta, score del feature fundus bilineal en la coordenada GT | recall **{m1_bilinear_recall * 100:.2f}%**; percentil **{m1_bilinear_percentile:.3f}%** |
| M2 | Pico duro del mapa de correlación exacto del modelo, distancia al centro GT, con cabeza de regresión omitida | ≤10 px **{m2_recall * 100:.1f}%**; distancia mediana **{m2_median:.2f} px** |
| M3 | Masa softmax del pico y correlación de Pearson punto-biserial con `M2 ≤10 px` | masa mediana **{np.median(masses):.6f}**; correlación **{fmt(m3_corr)}** |

La correlación del modelo no está condicionada por la rotación ni escala que produce su cabeza;
por eso “dar θ/s GT” equivale aquí a omitir completamente esa cabeza y evaluar el pico contra el
centro calculado directamente con la matriz GT. No se modificó el mapa de correlación.

## ¿Reproduce 28.6%?

{reproduction}

La comparación no es equivalente al fold real de entrenamiento: el disco local solo contiene cinco
casos de Mock Test y un único escenario, de modo que `GroupKFold(5)` es imposible. El 28.6% y el
{m2_recall * 100:.1f}% deben conservarse ambos; este último es la referencia local reproducible.

## M3 por caso

| Caso | Distancia M2 (px) | Masa del pico | Correcto ≤10 px |
|---|---:|---:|---:|
{rows}

## Gate propuesto para T2-R10

- **M1:** recall top-1% ≥ **{gate_m1 * 100:.1f}%** (y percentil de rank mediano menor que **{m1_percentile:.3f}%**).
- **M2:** recall ≤10 px ≥ **{gate_m2 * 100:.1f}%** y distancia mediana menor que **{m2_median:.2f} px**.
- **M3:** correlación confianza/corrección ≥ **{gate_m3:.2f}**. Si M2 no contiene ambas clases,
  M3 queda no identificable y debe repetirse sobre un fold con varios escenarios.

Los márgenes de recall exigen al menos +10 puntos porcentuales sobre el baseline local, con el piso
pre-registrado de 50% de la literatura. Dado `n=5`, estos umbrales son provisionales y deben aprobarse
o recalcularse al disponer del fold real.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_rows = [
        {"case_id": case, "scenario": case.split("/", 1)[0], "condition": "paired",
         "center_distance_px": float(distance), "peak_mass": float(mass),
         "correct_10px": bool(ok)}
        for case, distance, mass, ok in per_case
    ]
    if args.oct_shuffle:
        if len(val_idx) < 2:
            raise RuntimeError("--oct-shuffle requires at least two validation cases")
        shuffled = build_oct_derangement(val_idx)
        shuffle_rng = np.random.default_rng(args.seed)
        shuffle_nearest, shuffle_distances = [], []
        for target_index, source_index in zip(val_idx, shuffled):
            target, source = dataset[int(target_index)], dataset[int(source_index)]
            fundus = target["fundus"].unsqueeze(0).to(args.device)
            enface = source["enface"].unsqueeze(0).to(args.device)
            gt_case = target["gt_matrix"].unsqueeze(0).to(args.device)
            with torch.inference_mode():
                resized = F.interpolate(enface, size=(int(SCALE_REF), int(SCALE_REF)),
                                        mode="bilinear", align_corners=False)
                fundus_feat = model.fundus_encoder(fundus)
                enface_feat = model.enface_encoder(resized)
                ranks, _, _ = dense_retrieval(fundus_feat, enface_feat, gt_case[0],
                                               args.k, shuffle_rng, fundus.shape[-2],
                                               fundus.shape[-1])
                shuffle_nearest.extend(ranks)
                logits = correlation_map(fundus_feat, enface_feat)
                peak = logits.flatten(1).argmax(dim=1)
                peak_y = torch.div(peak, logits.shape[-1], rounding_mode="floor")
                peak_x = peak % logits.shape[-1]
                peak_center = torch.stack([
                    (peak_x.float() - 0.5) * (fundus.shape[-1] / fundus_feat.shape[-1]),
                    (peak_y.float() - 0.5) * (fundus.shape[-2] / fundus_feat.shape[-2]),
                ], dim=1)
                true_center = gt_case[:, :2, :2] @ torch.tensor(
                    [0.5, 0.5], dtype=gt_case.dtype, device=gt_case.device) + gt_case[:, :2, 2]
                distance = torch.linalg.vector_norm(peak_center - true_center, dim=1).item()
                mass = torch.softmax(logits.flatten(1), dim=1).max(dim=1).values.item()
            shuffle_distances.append(distance)
            case_id = f"{target['scenario']}/{target['frame_id']}"
            source_id = f"{source['scenario']}/{source['frame_id']}"
            jsonl_rows.append({"case_id": case_id, "scenario": target["scenario"],
                               "condition": "oct_shuffle", "oct_source_case_id": source_id,
                               "center_distance_px": float(distance), "peak_mass": float(mass),
                               "correct_10px": bool(distance <= 10.0)})
        shuffle_distances = np.asarray(shuffle_distances)
        shuffle_m1 = float(np.mean(np.asarray(shuffle_nearest) <= top_count))
        shuffle_m2 = float(np.mean(shuffle_distances <= 10.0))
        report += ("\n\n## Control OCT-shuffle\n\n"
                   "Mismo fold y GT; únicamente el OCT se rota un caso, sin auto-pares.\n\n"
                   "| Métrica | Paired | OCT-shuffle | Delta |\n|---|---:|---:|---:|\n"
                   f"| M1 top-1% | {m1_recall:.6f} | {shuffle_m1:.6f} | {m1_recall-shuffle_m1:.6f} |\n"
                   f"| M2 <=10 px | {m2_recall:.6f} | {shuffle_m2:.6f} | {m2_recall-shuffle_m2:.6f} |\n"
                   f"| M2 mediana px | {m2_median:.3f} | {np.median(shuffle_distances):.3f} | "
                   f"{m2_median-np.median(shuffle_distances):.3f} |\n")
    args.output.write_text(report, encoding="utf-8")
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_case_jsonl(args.jsonl, jsonl_rows)

    print("\nRESUMEN")
    print(f"data_root={root} cases={len(val_idx)} groups={unique_groups} fallback={used_fallback}")
    print(f"M1 nearest_top1pct={m1_recall:.6f} nearest_median_rank_pct={m1_percentile:.6f}")
    print(f"M1 bilinear_top1pct={m1_bilinear_recall:.6f} bilinear_median_rank_pct={m1_bilinear_percentile:.6f}")
    print(f"M2 recall_10px={m2_recall:.6f} median_distance_px={m2_median:.6f}")
    print(f"M3 median_peak_mass={np.median(masses):.9f} corr_mass_correct={fmt(m3_corr, 6)}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
