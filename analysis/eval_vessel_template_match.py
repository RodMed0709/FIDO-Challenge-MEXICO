#!/usr/bin/env python3
"""Evalúa el motor de matching clásico de vasos (T2-R4, Motor 1:
`fido.models.vessel_template_match.match_vessels`) sobre el Mock Test local
de Task 2 (5 casos, 1 escenario).

Usa las máscaras GT de vaso del fundus directamente (`arteriesorveins.png`,
`mask > 0`), NO la salida de un segmentador — el segmentador (Parte 1 de
T2-R4) todavía no está entrenado (sin GPU disponible ahora). El objetivo de
este script es aislar y verificar el MÉTODO DE MATCHING en sí: si funciona
razonablemente con la máscara GT como entrada "perfecta", vale la pena
entrenar el segmentador; si ni siquiera con GT funciona, el problema está en
el matching, no en la segmentación.

100% CPU (numpy + OpenCV), no necesita GPU ni el pod. Correr con:

    python analysis/eval_vessel_template_match.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fido.data.common import enface_vessel_density, load_label_map, load_volume_label_maps  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402
from fido.geometry import compose_similarity, corner_auc, corner_error  # noqa: E402
from fido.models.vessel_template_match import match_vessels  # noqa: E402

MOCK_TEST_ROOT = PROJECT_ROOT / "data" / "Mock Test" / "Task 2"


def main() -> None:
    cases = find_task2_cases(MOCK_TEST_ROOT)
    if not cases:
        raise SystemExit(f"No se encontraron casos en {MOCK_TEST_ROOT}")
    print(f"{len(cases)} casos encontrados en {MOCK_TEST_ROOT}\n")

    errors = []
    for case in cases:
        scenario_dir = case["scenario_dir"]
        frame_id = case["frame_id"]

        # Máscara de vaso del fundus, GT directo (sin segmentador todavía).
        vessel_mask_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
        fundus_vessel_gt = (load_label_map(vessel_mask_path) > 0).astype(np.float32)

        # Densidad de vaso del en-face, ya calculada del volumen OCT (T2-R3:
        # clase ArteriesOrVeins=3, confirmada limpia).
        volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
        seg_volume = load_volume_label_maps(volume_dir / "Segmentation")
        enface_vessel = enface_vessel_density(seg_volume)

        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)

        t0 = time.time()
        try:
            params = match_vessels(fundus_vessel_gt, enface_vessel)
        except RuntimeError as exc:
            # Root cause investigado (ver analysis/verify_vessel_template_match_synthetic.py):
            # ninguna celda de la grilla produce un patch válido cuando
            # enface_vessel es enteramente cero (sin clase ArteriesOrVeins en
            # ese volumen OCT concreto) -- caso genuinamente degenerado, no un
            # bug de geometría. Se salta el caso y se continúa el batch en vez
            # de tronar todo el script.
            print(f"[{case['scenario']}/{frame_id}] SALTADO: {exc}")
            continue
        elapsed = time.time() - t0

        pred_matrix = compose_similarity(
            params["tx"], params["ty"], params["cos_theta"], params["sin_theta"], params["scale"], reflect=True
        )
        error = float(corner_error(pred_matrix, gt_matrix))
        errors.append(error)

        print(
            f"[{case['scenario']}/{frame_id}] corner_error={error:7.2f}px  "
            f"match_score={params['match_score']:.3f}  scale={params['scale']:6.1f}  "
            f"({elapsed:.1f}s)"
        )

    errors_arr = np.array(errors)
    auc = corner_auc(errors_arr)
    print()
    print(f"corner_error: media={errors_arr.mean():.2f}px  min={errors_arr.min():.2f}px  max={errors_arr.max():.2f}px")
    print(f"corner_auc (umbrales 0..10px, igual que el scoring oficial): {auc:.4f}")
    print("Referencia: T2-R2 (baseline sin vasos, smoke test sin entrenar de verdad) ~165-200px de error.")


if __name__ == "__main__":
    main()
