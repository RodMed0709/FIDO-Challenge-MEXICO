"""Cobertura real del instrumento como landmark cross-modal en Task 2.

Responde, solo a partir de las anotaciones JSON (sin tocar el volumen OCT):
  1. En cuantos de los 1214 casos de train el keypoint de instrumento del
     fundus (Endgripping Forceps / Endoilluminator), proyectado por la matriz
     GT a coordenadas uv, cae dentro de la huella en-face [0,1]x[0,1].
  2. Cuantos puntos por caso (0/1/2/3) -- el techo de DOF resolubles por
     Procrustes/Umeyama en cierre cerrado (2 puntos -> 4 DOF).
  3. Desglose por escenario (la cobertura es una propiedad de escena, no
     uniforme por caso).

Reusa `points_from_group`, `inside_uv` de
`analysis/verify_task2_enface_convention.py` para no reimplementar la
proyeccion fundus->uv (misma convencion congelada `transpose__flip_u__flip_v`).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.verify_task2_enface_convention import inside_uv, points_from_group  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True,
                         help="Directorio con Scenario_NN/*.json (solo anotaciones, sin volumen).")
    args = parser.parse_args()

    total = 0
    with_forceps_annot = 0
    with_illum_annot = 0
    with_inside = 0
    per_case_counts: list[int] = []
    per_scenario: dict[str, dict[str, int]] = {}

    for scenario_dir in sorted(args.annotations_root.glob("Scenario_*")):
        stats = per_scenario.setdefault(scenario_dir.name, {"total": 0, "inside": 0})
        for json_path in sorted(scenario_dir.glob("*.json")):
            total += 1
            stats["total"] += 1
            data = json.loads(json_path.read_text(encoding="utf-8"))
            matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
            forceps = points_from_group(data, "Endgripping Forceps",
                                        ("Right Head Tip", "Left Head Tip", "Joint Tip", "Start"))
            illum = points_from_group(data, "Endoilluminator", ("Tip", "Start"))
            if len(forceps):
                with_forceps_annot += 1
            if len(illum):
                with_illum_annot += 1
            points = (np.concatenate([forceps, illum]) if (len(forceps) or len(illum))
                     else np.empty((0, 2)))
            inside = inside_uv(points, matrix)
            per_case_counts.append(len(inside))
            if len(inside):
                with_inside += 1
                stats["inside"] += 1

    print(f"casos totales (json con GT Task 2): {total}")
    print(f"casos con anotacion de forceps (cualquiera): {with_forceps_annot}")
    print(f"casos con anotacion de iluminador (cualquiera): {with_illum_annot}")
    print(f"casos con >=1 keypoint de instrumento DENTRO de la huella uv: "
         f"{with_inside}/{total} ({with_inside/total:.4f})")
    print(f"suma de puntos dentro de la huella (todos los casos): {sum(per_case_counts)}")

    counts = collections.Counter(per_case_counts)
    print("distribucion de puntos-dentro-de-huella por caso:")
    for k in sorted(counts):
        print(f"  {k} puntos: {counts[k]} casos")
    n_ge2 = sum(v for k, v in counts.items() if k >= 2)
    print(f"casos con >=2 puntos (Procrustes/Umeyama cerrado, 4 DOF): "
         f"{n_ge2}/{total} ({n_ge2/total:.4f})")
    n_eq1 = counts.get(1, 0)
    print(f"casos con ==1 punto (solo traslacion, requiere prior externo de rotacion/escala): "
         f"{n_eq1}/{total} ({n_eq1/total:.4f})")

    print("\ndesglose por escenario (inside/total):")
    for name, stats in sorted(per_scenario.items()):
        frac = stats["inside"] / stats["total"] if stats["total"] else float("nan")
        print(f"  {name}: {stats['inside']}/{stats['total']} = {frac:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
