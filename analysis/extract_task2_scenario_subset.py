#!/usr/bin/env python3
"""Extrae del zip de un escenario de Task 2 solo lo necesario para el
experimento 102 (búsqueda densa por solapamiento de vasos):

- Numerical/<frame>.json
- Stereo Left/<frame>/microscope.png
- Stereo Left/<frame>/Segmentation/arteriesorveins.png
- iOCT Microscope/Volume/<frame>/Segmentation/*.png (128 slices, todas las
  clases en un solo label map por slice)

Evita extraer Canvas/, Bscan/, visibility.png, el volumen RAW en escala de
grises, y las otras máscaras del lado fundus (cannula/forceps/endoilluminator/
ilm) que no hacen falta para este experimento. Reduce ~10x el tiempo/disco
frente a `unzip` completo.

Uso:
    python analysis/extract_task2_scenario_subset.py Scenario_07 Scenario_10
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = PROJECT_ROOT / "data" / "Task 2"
OUT_DIR = PROJECT_ROOT / "data" / "Task 2"


def wanted(name: str) -> bool:
    if name.startswith("Numerical/") and name.endswith(".json"):
        return True
    if "Stereo Left" in name and name.endswith("microscope.png"):
        return True
    if "Stereo Left" in name and name.endswith("Segmentation/arteriesorveins.png"):
        return True
    if "iOCT Microscope/Volume" in name and "/Segmentation/" in name and name.endswith(".png"):
        return True
    return False


def main() -> None:
    scenarios = sys.argv[1:]
    if not scenarios:
        raise SystemExit("Uso: extract_task2_scenario_subset.py Scenario_NN [Scenario_MM ...]")

    for scenario in scenarios:
        zip_path = ZIP_DIR / f"{scenario}.zip"
        if not zip_path.exists():
            print(f"[{scenario}] SALTADO: no existe {zip_path}")
            continue
        out_scenario_dir = OUT_DIR / scenario
        if out_scenario_dir.exists():
            print(f"[{scenario}] SALTADO: {out_scenario_dir} ya existe")
            continue

        print(f"[{scenario}] abriendo {zip_path} ...")
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if wanted(n)]
            print(f"[{scenario}] extrayendo {len(names)} archivos a {out_scenario_dir} ...")
            z.extractall(path=out_scenario_dir, members=names)
        print(f"[{scenario}] listo.")


if __name__ == "__main__":
    main()
