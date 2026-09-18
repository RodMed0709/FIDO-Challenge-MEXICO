#!/usr/bin/env python3
"""T1-93: distribucion real del target oficial de distancia (px) de Task 1.

Necesaria para elegir `reg_max`/`[d_min, d_max]` de la cabeza DFL a partir de
DATOS, no copiando el `-1mm..6mm` del paper de los organizadores (unidades y
sensor distintos -- ver `experiments/87-organizer-baseline/INFORME.md`
Sec.2.2 y 8B). Target = `gt[2] / GROUND_TRUTH_DISTANCE_SCALE` con
`GROUND_TRUTH_DISTANCE_SCALE = 7.8125` (mismo calculo exacto que
`Task1Dataset._load_case` en `src/fido/data/task1.py`), sobre TODOS los
casos con canula activa de los 10 escenarios locales
(`data/_annotations/Task 1/`, ~61,691 JSON, no hace falta tocar los zips).

    python analysis/measure_task1_distance_target_histogram.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_DISTANCE_SCALE = (4.0 / 512) * 1000  # 7.8125, igual a src/fido/data/task1.py


def cannula_active(annotation: dict) -> bool:
    ilm = annotation.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
    return ilm is not None and ilm < 1e6


def main() -> None:
    annotations_root = ROOT / "data" / "_annotations" / "Task 1"
    output_path = ROOT / "experiments" / "93-t1-dfl" / "DISTANCE_TARGET_HISTOGRAM.md"

    targets = []
    per_scenario_counts: dict[str, int] = {}
    n_total_frames = n_active = n_missing_gt = 0
    started = time.time()

    for scen_dir in sorted(p for p in annotations_root.iterdir() if p.is_dir()):
        scenario_targets = []
        for json_path in sorted(scen_dir.glob("*.json")):
            n_total_frames += 1
            try:
                annotation = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not cannula_active(annotation):
                continue
            n_active += 1
            gt = annotation.get("Ground Truth", {}).get("Task 1")
            if gt is None or len(gt) < 3:
                n_missing_gt += 1
                continue
            scenario_targets.append(float(gt[2]) / GROUND_TRUTH_DISTANCE_SCALE)
        per_scenario_counts[scen_dir.name] = len(scenario_targets)
        targets.extend(scenario_targets)
        print(f"[{time.time()-started:6.1f}s] {scen_dir.name}: {len(scenario_targets):,} "
              f"targets validos (canula activa)", flush=True)

    targets_array = np.asarray(targets, dtype=np.float64)
    n = targets_array.size
    percentile_levels = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    percentiles = {p: float(np.percentile(targets_array, p)) for p in percentile_levels}
    n_negative = int((targets_array < 0).sum())

    lines = [
        "# T1-93 -- distribucion real del target oficial de distancia (px)",
        "",
        f"**Fuente:** `data/_annotations/Task 1/Scenario_01..10` "
        f"(`gt[2] / {GROUND_TRUTH_DISTANCE_SCALE}`, `gt = annotation[\"Ground Truth\"][\"Task 1\"]`).  ",
        f"**Filtro:** solo canula activa (`ILM Distance` presente y `< 1e6`), igual que "
        f"`_cannula_is_active` en `src/fido/data/task1.py`.  ",
        f"**Frames totales escaneados:** {n_total_frames:,}  ",
        f"**Frames con canula activa:** {n_active:,}  ",
        f"**Frames con canula activa pero sin `Ground Truth.Task 1`:** {n_missing_gt:,}  ",
        f"**n usado para el histograma:** {n:,}  ",
        f"**Tiempo total:** {time.time() - started:.1f}s  ",
        "",
        f"min={targets_array.min():.3f}px  max={targets_array.max():.3f}px  "
        f"media={targets_array.mean():.3f}px  std={targets_array.std():.3f}px  ",
        f"**Casos con target negativo:** {n_negative:,} ({n_negative / n:.4%})  " if n else "",
        "",
        "## Percentiles",
        "",
        "| percentil | valor (px) |",
        "|---:|---:|",
    ]
    for p in percentile_levels:
        lines.append(f"| {p} | {percentiles[p]:.3f} |")
    lines += [
        "",
        "## Por escenario (n con target valido)",
        "",
        "| escenario | n |",
        "|---|---:|",
    ]
    for scenario, count in per_scenario_counts.items():
        lines.append(f"| {scenario} | {count:,} |")
    lines += [
        "",
        "## Reproduccion",
        "",
        "```bash",
        "python analysis/measure_task1_distance_target_histogram.py",
        "```",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
