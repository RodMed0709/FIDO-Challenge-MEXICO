#!/usr/bin/env python3
"""Saca solo los JSON de anotaciones de los zips, sin tocar las imágenes.

Los zips del dataset pesan decenas de GB porque van llenos de PNG, pero las
anotaciones son unos pocos MB. Para todo el análisis de geometría (estructura de
la matriz, distribuciones, presupuesto de error) basta con los JSON, y
extraerlos así evita descomprimir 83 GB.

    python analysis/extract_annotations.py --task "Task 2"
    python analysis/extract_annotations.py --task "Task 1"
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Task 2", help='"Task 1" o "Task 2"')
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    source = args.data / args.task
    destination = args.data / "_annotations" / args.task
    zips = sorted(source.glob("Scenario_*.zip"))
    if not zips:
        raise SystemExit(f"No hay zips en {source}")

    total = 0
    for zip_path in zips:
        scenario = zip_path.stem
        out_dir = destination / scenario
        out_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as archive:
            # The zips have the scenario contents at the root, so the numerical
            # annotations live directly under "Numerical/".
            members = [n for n in archive.namelist()
                       if n.startswith("Numerical/") and n.endswith(".json")]
            for member in members:
                (out_dir / Path(member).name).write_bytes(archive.read(member))

        print(f"  {scenario}: {len(members)} JSONs")
        total += len(members)

    print(f"\nTotal: {total} snapshots -> {destination}")


if __name__ == "__main__":
    main()
