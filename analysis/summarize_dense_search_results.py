#!/usr/bin/env python3
"""Agrega los resultados de `dense_search_task2_vessels.py` en las métricas
que decide el experimento 102: corner-AUC oficial por método, distribución
de error, fracción dentro de 10/20/50px, y percentil de la puntuación GT.

Uso:
    python analysis/summarize_dense_search_results.py \
        experiments/102-t2-dense-search/results_train.json \
        experiments/102-t2-dense-search/results_mock.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fido.geometry import corner_auc  # noqa: E402

METHODS = ("ncc", "iou", "dice", "chamfer", "chamfer_sym")


def summarize(records: list[dict], label: str) -> None:
    ok_records = [r for r in records if r.get("status") == "ok"]
    print(f"\n=== {label}: {len(ok_records)}/{len(records)} casos OK ===")
    if not ok_records:
        return
    elapsed = np.array([r.get("elapsed_s", np.nan) for r in ok_records])
    print(f"tiempo/caso: media={elapsed.mean():.1f}s mediana={np.median(elapsed):.1f}s "
          f"min={elapsed.min():.1f}s max={elapsed.max():.1f}s")

    for method in METHODS:
        entries = [r[method] for r in ok_records if r.get(method, {}).get("status") == "ok"]
        if not entries:
            print(f"  {method:12s}: sin resultados válidos")
            continue
        errors = np.array([e["corner_error"] for e in entries])
        auc = corner_auc(errors)
        pcts = [e["gt_score_percentile_vs_cell_maxima"] for e in entries
                if e["gt_score_percentile_vs_cell_maxima"] is not None]
        pcts = np.array(pcts) if pcts else np.array([np.nan])
        frac10 = float(np.mean(errors <= 10))
        frac20 = float(np.mean(errors <= 20))
        frac50 = float(np.mean(errors <= 50))
        print(f"  {method:12s}: n={len(entries):4d}  corner_AUC={auc:.4f}  "
              f"median_err={np.median(errors):7.1f}px  mean_err={errors.mean():7.1f}px  "
              f"p10={np.percentile(errors,10):6.1f}  p90={np.percentile(errors,90):6.1f}  "
              f"<=10px={frac10:.3f}  <=20px={frac20:.3f}  <=50px={frac50:.3f}  "
              f"gt_score_percentile: media={np.nanmean(pcts):.1f} mediana={np.nanmedian(pcts):.1f}")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("Uso: summarize_dense_search_results.py results1.json [results2.json ...]")
    for path in paths:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        summarize(records, Path(path).stem)


if __name__ == "__main__":
    main()
