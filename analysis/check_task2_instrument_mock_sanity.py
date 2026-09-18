"""Sanity check de un solo uso sobre Mock Test (n=5, Scenario_12): reusa
`measure_case` de `analysis/verify_task2_enface_convention.py` tal cual (mismo
codigo que corrio en train) para ver si el keypoint de instrumento del fundus,
proyectado por la matriz GT, cae cerca de la segmentacion REAL de instrumento
en el volumen OCT -- no solo dentro de la huella uv [0,1]x[0,1].

Uso diagnostico, no de seleccion de hipotesis ni de tuning (CONSTITUTION.md
regla 3): un solo calculo de lectura, n=5, para acotar el riesgo de
generalizacion del rung 89 antes de comprometer trabajo de GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from analysis.verify_task2_enface_convention import measure_case  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                         help='Ej. "data/Mock Test/Task 2"')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = find_task2_cases(args.root)
    print(f"casos encontrados: {len(cases)}")
    for index, case in enumerate(cases):
        result = measure_case(case, index, args.seed)
        case_id = result["case_id"]
        if result["error"]:
            print(f"{case_id}: error={result['error']}")
            continue
        c_count = result.get("c_count", 0)
        a_count = result.get("a_count", 0)
        print(f"{case_id}: a_count(vasos)={a_count} c_count(instrumento, dentro de huella)={c_count}")
        if result.get("c"):
            observed = result["c"]["transpose__flip_u__flip_v"]["observed"]
            print(f"  distancias observadas (fraccion del lado del cuadrado unitario): "
                 f"{sorted(float(v) for v in observed)}")
            print("  (radios usados en el oraculo train: 0.005 / 0.01 / 0.02 / 0.04)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
