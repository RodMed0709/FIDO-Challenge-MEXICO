#!/usr/bin/env python3
"""Escaneo paralelo (ThreadPoolExecutor, igual que
`fido.data.task1.find_task1_cases`) de `data/_annotations/Task 1/` para
construir la lista de casos con canula activa, con progreso impreso."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CANNULA_INACTIVE_SENTINEL = 1e6


def cannula_active(json_path: Path) -> bool:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    ilm = data.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
    return ilm is not None and ilm < CANNULA_INACTIVE_SENTINEL


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/_annotations/Task 1")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "experiments/101-t1-tta-ensemble/cases_cache.json")

    cases = []
    t0 = time.time()
    for scenario_dir in sorted(root.glob("Scenario_*")):
        json_paths = sorted(scenario_dir.glob("*.json"))
        with ThreadPoolExecutor(max_workers=16) as pool:
            flags = list(pool.map(cannula_active, json_paths))
        active = [p.stem for p, f in zip(json_paths, flags) if f]
        cases.extend({"scenario": scenario_dir.name, "frame_id": frame_id} for frame_id in active)
        print(f"  {scenario_dir.name}: {len(active)}/{len(json_paths)} activos "
              f"({time.time() - t0:.1f}s acumulados)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases), encoding="utf-8")
    print(f"TOTAL: {len(cases)} casos en {time.time() - t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
