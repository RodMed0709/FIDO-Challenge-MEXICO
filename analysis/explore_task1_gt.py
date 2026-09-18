#!/usr/bin/env python3
"""Explora el ground truth de Task 1: keypoints y distancia herramienta-tejido.

El score de Task 1 es 0.7 x keypoint_AUC + 0.3 x distance_AUC. La componente de
distancia pesa un tercio y es la menos obvia de las dos, así que vale la pena
saber cómo se distribuye antes de diseñar nada.

Ojo con la escala: el GT almacena la distancia x10 y el scoring la divide entre
10 antes de comparar. Aquí se reporta ya dividida, o sea en las mismas unidades
en que debe predecir el modelo.

    python analysis/explore_task1_gt.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_DISTANCE_SCALE = (4.0 / 512) * 1000  # = 7.8125, igual que en scoring_keypoints.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path,
                        default=PROJECT_ROOT / "data" / "_annotations" / "Task 1")
    parser.add_argument("--limit", type=int, default=0,
                        help="Analizar solo los primeros N por escenario (0 = todos)")
    args = parser.parse_args()

    scenario_dirs = sorted(d for d in args.annotations.iterdir() if d.is_dir())
    if not scenario_dirs:
        raise SystemExit(f"No hay escenarios en {args.annotations}")

    xs, ys, distances, per_scenario = [], [], [], Counter()
    missing = 0

    for scenario_dir in scenario_dirs:
        paths = sorted(scenario_dir.glob("*.json"))
        if args.limit:
            paths = paths[: args.limit]
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            ground_truth = data.get("Ground Truth", {}).get("Task 1")
            if ground_truth is None:
                missing += 1
                continue
            xs.append(float(ground_truth[0]))
            ys.append(float(ground_truth[1]))
            distances.append(float(ground_truth[2]) / GROUND_TRUTH_DISTANCE_SCALE)
            per_scenario[scenario_dir.name] += 1

    x = np.asarray(xs)
    y = np.asarray(ys)
    distance = np.asarray(distances)

    print(f"Frames con ground truth : {len(x):,}")
    print(f"Frames sin ground truth : {missing:,}")
    print(f"Escenarios              : {len(per_scenario)}")
    print()

    print("Keypoint (pixeles de la imagen de 1024x1024):")
    for name, values in (("x", x), ("y", y)):
        print(f"  {name}  media {values.mean():9.2f}  sd {values.std():8.2f}"
              f"   min {values.min():10.2f}   max {values.max():10.2f}")
    inside = ((x >= 0) & (x < 1024) & (y >= 0) & (y < 1024))
    print(f"  dentro del cuadro 1024x1024 : {100 * inside.mean():6.2f} %")
    print()

    # A constant predictor is the honest floor: if predicting the mean keypoint
    # already scores well, the task is easier than it looks.
    centre = np.array([x[inside].mean(), y[inside].mean()])
    errors = np.linalg.norm(np.stack([x, y], axis=1) - centre, axis=1)
    print(f"  error si siempre predijeramos la media {np.round(centre, 1).tolist()}:")
    print(f"    media {errors.mean():.1f} px, mediana {np.median(errors):.1f} px")
    print()

    print("Distancia herramienta-tejido (ya dividida entre 10):")
    finite = np.isfinite(distance)
    # The annotations use a huge sentinel (~2.147e9, i.e. int32 max) to mean
    # "no contact / not measurable". Treating it as a real number would poison
    # every statistic and every loss.
    sentinel = distance > 1e6
    print(f"  valores centinela (>1e6, 'sin medida') : {sentinel.sum():,}"
          f"  ({100 * sentinel.mean():.2f} %)")
    real = distance[finite & ~sentinel]
    if real.size:
        print(f"  reales: n={real.size:,}  media {real.mean():8.3f}  sd {real.std():8.3f}"
              f"   min {real.min():8.3f}   max {real.max():8.3f}")
        for q in (1, 5, 25, 50, 75, 95, 99):
            print(f"    p{q:<2d} = {np.percentile(real, q):9.3f}")

        # distance_AUC uses the same 0..10 integer thresholds as the keypoints,
        # so a constant prediction is worth measuring.
        def auc(values, max_threshold=10):
            return float(np.mean([(values <= t).mean() for t in range(max_threshold + 1)]))

        print()
        best_constant, best_auc = None, -1.0
        for candidate in np.percentile(real, np.arange(1, 100)):
            score = auc(np.abs(real - candidate))
            if score > best_auc:
                best_constant, best_auc = candidate, score
        print(f"  mejor prediccion CONSTANTE de distancia: {best_constant:.3f}"
              f"  -> distance_AUC = {best_auc:.4f}")
        print(f"  contribucion al score final (peso 0.3) = {0.3 * best_auc:.4f}")


if __name__ == "__main__":
    main()
