"""Estadística de las anotaciones de Task 2 — solo lee JSON, no toca imágenes.

Responde tres preguntas que el diseño del modelo necesita y que nadie midió:
  1. Distribución de los 4 puntos del crosshair y de los 4 DOF derivados.
  2. Si la escala de train y la del Mock Test son rangos disjuntos.
  3. Qué instrumentos aparecen y cuáles tienen keypoint dentro de la huella OCT.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_cases(root: Path):
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        numerical = scenario / "Numerical"
        if not numerical.is_dir():
            continue
        for path in sorted(numerical.glob("*.json")):
            try:
                yield scenario.name, path.stem, json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:                      # MooseFS/JSON roto: se salta
                print(f"  saltado {scenario.name}/{path.stem}: {exc}")


def describe(name, values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return f"{name:<22} n=0"
    return (f"{name:<22} n={a.size:<6} media={a.mean():8.2f}  sd={a.std():7.2f}  "
            f"min={a.min():8.2f}  p10={np.percentile(a,10):8.2f}  "
            f"p90={np.percentile(a,90):8.2f}  max={a.max():8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    scales, angles, tx, ty, dets = [], [], [], [], []
    tools, tools_with_kp = Counter(), Counter()
    n_vessel_kp, n_cases = [], 0

    for scenario, frame, data in load_cases(args.root):
        gt = (data.get("Ground Truth") or {}).get("Task 2")
        if gt is None:
            continue
        n_cases += 1
        m = np.asarray(gt, dtype=float)
        A, t = m[:2, :2], m[:2, 2]
        dets.append(float(np.linalg.det(A)))
        # escala = norma media de las columnas; angulo = orientacion de la 1a columna
        scales.append(float((np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1])) / 2))
        angles.append(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))))
        tx.append(float(t[0])); ty.append(float(t[1]))

        for tool in (data.get("Surgical Tool") or {}):
            tools[tool] += 1
        kp = data.get("Keypoints") or {}
        for group, points in kp.items():
            if group in ("Vasculature", "iOCT Microscope Crosshair"):
                continue
            if isinstance(points, dict) and any(v is not None for v in points.values()):
                tools_with_kp[group] += 1
        vasc = kp.get("Vasculature") or {}
        n_vessel_kp.append(sum(1 for v in vasc.values() if v is not None))

    tag = f" [{args.label}]" if args.label else ""
    print(f"\n===== Task 2 annotation stats{tag} — root={args.root} =====")
    print(f"casos con GT: {n_cases}")
    print(describe("escala (px)", scales))
    print(describe("angulo (deg)", angles))
    print(describe("tx", tx)); print(describe("ty", ty))
    print(describe("det(A)", dets))
    print(f"det negativo: {sum(1 for d in dets if d < 0)}/{len(dets)}")
    print(describe("n keypoints vasc", n_vessel_kp))
    print(f"instrumentos en 'Surgical Tool': {dict(tools)}")
    print(f"grupos de keypoints de instrumento: {dict(tools_with_kp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
