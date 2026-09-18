#!/usr/bin/env python3
"""¿La matriz de Task 2 es de verdad una similitud reflejada de 4 DOF?

El diseño de Task 2 apuesta a que el ground truth tiene solo 4 grados de
libertad (tx, ty, θ, s) en vez de los 6 de una afín general. Esa conclusión salió
de 5 frames de un solo escenario del Mock Test — muy poca evidencia para
construir un modelo encima. Este script la reverifica sobre todos los snapshots
de entrenamiento disponibles.

Entrada: los JSON de anotaciones, extraídos de los zips sin descomprimir las
imágenes (ver `extract_annotations.py`).

    python analysis/verify_task2_structure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "_annotations" / "Task 2"


def describe(name: str, values: np.ndarray) -> str:
    return (f"  {name:8s} media {values.mean():10.3f}  sd {values.std():9.3f}"
            f"   min {values.min():10.3f}   max {values.max():10.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()

    paths = sorted(args.annotations.rglob("*.json"))
    if not paths:
        raise SystemExit(f"No hay JSONs en {args.annotations}")

    scenarios, matrices, missing = [], [], 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        ground_truth = data.get("Ground Truth", {}).get("Task 2")
        if ground_truth is None:
            missing += 1
            continue
        scenarios.append(path.parent.name)
        matrices.append(np.asarray(ground_truth, dtype=np.float64))

    stack = np.stack(matrices)                      # (N, 3, 3)
    linear = stack[:, :2, :2]                       # (N, 2, 2)
    translation = stack[:, :2, 2]                   # (N, 2)

    col0, col1 = linear[:, :, 0], linear[:, :, 1]
    norm0 = np.linalg.norm(col0, axis=1)
    norm1 = np.linalg.norm(col1, axis=1)
    ratio = norm0 / norm1
    cosine = np.einsum("ij,ij->i", col0, col1) / (norm0 * norm1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    determinant = np.linalg.det(linear)
    theta = np.degrees(np.arctan2(col0[:, 1], col0[:, 0]))

    print(f"Snapshots con ground truth : {len(matrices)}")
    print(f"Snapshots sin ground truth : {missing}")
    print(f"Escenarios                 : {len(set(scenarios))}")
    print()

    print("Escala y ortogonalidad:")
    print(describe("|col0|", norm0))
    print(describe("|col1|", norm1))
    print(describe("razon", ratio))
    print(describe("angulo", angle))
    print()

    print("Traslacion y rotacion:")
    print(describe("tx", translation[:, 0]))
    print(describe("ty", translation[:, 1]))
    print(describe("theta", theta))
    print()

    print("Pruebas de la hipotesis de similitud reflejada:")
    print(f"  determinante negativo        : {100 * (determinant < 0).mean():6.2f} %")
    print(f"  |razon - 1| < 0.05           : {100 * (np.abs(ratio - 1) < 0.05).mean():6.2f} %")
    print(f"  |angulo - 90 deg| < 5 deg    : {100 * (np.abs(angle - 90) < 5).mean():6.2f} %")
    print(f"  |angulo - 90 deg| < 10 deg   : {100 * (np.abs(angle - 90) < 10).mean():6.2f} %")
    print()

    # The decisive test: how much score would we lose by *forcing* the 4-DOF
    # form? Fit the closest reflected similarity to each true matrix and measure
    # the resulting corner error against the real scoring metric. If that error
    # is small next to the 10 px threshold, the constraint is free.
    corners = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    def project(matrix: np.ndarray) -> np.ndarray:
        out = (matrix @ corners.T).T
        return out[:, :2] / out[:, 2:3]

    errors = []
    for matrix in matrices:
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        # Closest reflected similarity [[p, q], [q, -p]] in the least-squares
        # sense: average the two independent estimates of each parameter.
        p = (a - d) / 2.0
        q = (b + c) / 2.0
        approx = np.array([[p, q, matrix[0, 2]],
                           [q, -p, matrix[1, 2]],
                           [0.0, 0.0, 1.0]])
        errors.append(np.linalg.norm(project(matrix) - project(approx), axis=1).mean())

    errors = np.asarray(errors)
    print("Costo de FORZAR la forma de 4 DOF (error de esquinas contra el GT real):")
    print(describe("error px", errors))
    for threshold in (1, 2, 5, 10):
        print(f"  error < {threshold:2d} px : {100 * (errors < threshold).mean():6.2f} %")

    def auc(values: np.ndarray, max_threshold: int = 10) -> float:
        return float(np.mean([(values <= t).mean() for t in range(max_threshold + 1)]))

    print()
    print(f"  AUC techo de un modelo perfecto restringido a 4 DOF: {auc(errors):.4f}")
    print("  (1.0 significa que la restriccion no cuesta nada)")


if __name__ == "__main__":
    main()
