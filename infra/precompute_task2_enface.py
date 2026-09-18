#!/usr/bin/env python3
"""Precomputa el cache de Task 2: proyeccion en-face + fundus, por caso.

Motivo (medido el 2026-08-17 en el pod): `Task2Dataset.__getitem__` abre los 128
PNG del volumen OCT por caso (21.5 MB) solo para promediarlos sobre el eje de
profundidad y quedarse con una imagen de 128x512. Sobre MooseFS eso son ~128
lecturas de red por muestra y fue lo que dejo la GPU al 0% durante horas.

La proyeccion en-face es una funcion PURA del volumen, que es estatico: no hay
ninguna razon para recalcularla en cada epoca, y menos en cada corrida. Se
calcula UNA vez y se guarda como `.npz` por caso:

    128 PNG (21.5 MB)  ->  enface uint8 128x512 (65 KB)   ~330x menos

El fundus RGB tambien se guarda en el mismo `.npz` para que el entrenamiento no
toque MooseFS en absoluto.

Uso:
    python precompute_task2_enface.py --root /workspace/data/Task2 \
        --out /root/data_cache/Task2_enface --workers 12
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/FIDO_CHALLENGE/src")

from fido.data.common import enface_projection, load_rgb, load_volume_slices  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402


def process_case(args_tuple) -> tuple[str, str, bool, str]:
    scenario, frame_id, scenario_dir_str, out_dir_str = args_tuple
    scenario_dir = Path(scenario_dir_str)
    out_path = Path(out_dir_str) / scenario / f"{frame_id}.npz"

    if out_path.exists():
        return (scenario, frame_id, True, "ya existia")

    try:
        volume = load_volume_slices(scenario_dir / "iOCT Microscope" / "Volume" / frame_id)
        enface = enface_projection(volume)  # uint8 (n_slices, ancho)
        fundus = load_rgb(scenario_dir / "Stereo Left" / frame_id / "microscope.png")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atomica: si el proceso muere a media escritura, un .npz
        # truncado se veria como "ya existia" en la siguiente corrida.
        # OJO: el temporal DEBE terminar en `.npz` -- `np.savez_compressed`
        # le concatena la extension al nombre si no la trae, asi que un
        # `.npz.tmp` acaba escrito como `.npz.tmp.npz` y el rename falla.
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        # Cache native data only. Task2Dataset owns the one canonicalization
        # point, preventing raw and cached paths from diverging or double-flipping.
        np.savez_compressed(tmp, enface=enface, fundus=fundus,
                            enface_convention="native_slice_lateral_v1")
        tmp.replace(out_path)
        return (scenario, frame_id, True, "ok")
    except Exception as exc:  # noqa: BLE001 - se reporta y se sigue con los demas
        return (scenario, frame_id, False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    t0 = time.time()
    cases = find_task2_cases(args.root)
    print(f"{len(cases)} casos encontrados en {time.time()-t0:.1f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payloads = [
        (c["scenario"], c["frame_id"], str(c["scenario_dir"]), str(args.out))
        for c in cases
    ]

    ok = 0
    fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_case, p) for p in payloads]
        for i, fut in enumerate(as_completed(futures), 1):
            scenario, frame_id, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FALLO {scenario}/{frame_id}: {msg}", flush=True)
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                eta = (len(payloads) - i) / max(rate, 1e-6)
                print(f"[{i}/{len(payloads)}] {rate:.1f} casos/s  ETA {eta/60:.1f} min",
                      flush=True)

    print(f"TERMINADO: {ok} ok, {fail} fallidos, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
