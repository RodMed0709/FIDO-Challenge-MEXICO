#!/usr/bin/env python3
"""Smoke test permanente de la geometría de `vessel_template_match.py`
(T2-R4, Motor 1) — investigación de root cause del crash + corner_error
enorme reportados al correr `analysis/eval_vessel_template_match.py` sobre
el Mock Test (ver ATTACK_LADDER.md T2-R4).

Dos partes independientes:

1. **Caso sintético autoconsistente** (aísla bugs de geometría de cualquier
   ruido real): construye un patrón de vaso sintético asimétrico, lo coloca
   en un fundus en blanco con `compose_similarity(reflect=True)` + la MISMA
   fórmula de warp que usa `_build_patch` internamente (misma convención
   u=col/w_e, v=row/h_e), y verifica que `match_vessels` recupera los
   parámetros casi exactos. Si esto falla, hay un bug real en
   `vessel_template_match.py` (orden de esquinas, convención fila/columna,
   recuperación de tx/ty desde `matchTemplate`, etc.).

2. **Oráculo de alineación real** (aísla señal de búsqueda): sobre los casos
   reales del Mock Test, usa los parámetros REALES del GT (no una búsqueda
   en grilla) para construir el patch de `enface_vessel` en la escala/ángulo
   correctos, y compara su correlación (NCC) contra la región del fundus
   donde el GT dice que debería caer. Si esta correlación "oráculo" ya es
   ~0 incluso con la respuesta correcta, el problema NO es la búsqueda ni la
   geometría — es que la señal de densidad de vaso del en-face no se parece
   al patrón de vaso del fundus ni siquiera alineada perfectamente.

Resultado de esta investigación (T2-R4, ver ATTACK_LADDER.md): (1) PASA con
error ~0.09px — la cadena interna (patch build + matchTemplate +
fit_closed_form_similarity) es correcta, no hay bug de geometría. (2) oracle
NCC en [-0.012, 0.046] sobre los 4 casos válidos del Mock Test — ruido puro,
no señal. Root cause de los resultados malos de T2-R4/Motor 1: el patrón de
`enface_vessel_density` (fracción de profundidad ocupada por la clase
ArteriesOrVeins) es extremadamente disperso (~4-6% de píxeles no-cero, valor
máximo 1/n_slices_profundidad) y no reproduce la topología continua del
árbol vascular que se ve en el fundus — es una limitación real de la
representación de señal, no un bug de código. NO se tocó
`src/fido/geometry.py` ni `vendor/fido/`.

100% CPU (numpy + OpenCV). Correr con:

    python analysis/verify_vessel_template_match_synthetic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fido.data.common import enface_vessel_density, load_label_map, load_volume_label_maps  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402
from fido.geometry import compose_similarity, corner_error, decompose_similarity, project_corners  # noqa: E402
from fido.models.vessel_template_match import _build_patch, match_vessels  # noqa: E402

MOCK_TEST_ROOT = PROJECT_ROOT / "data" / "Mock Test" / "Task 2"


def run_synthetic_self_consistency_check() -> bool:
    """Parte 1: ver docstring del módulo. Devuelve True si pasa (<2px)."""
    print("=== 1. Caso sintético autoconsistente ===")
    h_e, w_e = 128, 512
    enface_vessel = np.zeros((h_e, w_e), dtype=np.float32)
    # Manchas asimétricas: cualquier mismatch de fila/columna o de orden de
    # esquinas rompe la simetría y produce un patrón claramente distinto.
    blobs = [(20, 60, 6), (100, 400, 10), (60, 250, 4), (15, 470, 5)]
    for (r, c, rad) in blobs:
        yy, xx = np.ogrid[:h_e, :w_e]
        enface_vessel[(yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2] = 1.0

    known = dict(tx=400.0, ty=550.0, scale=150.0, angle_deg=40.0)
    angle_rad = np.radians(known["angle_deg"])
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
    linear = known["scale"] * np.array([[cos_t, sin_t], [sin_t, -cos_t]], dtype=np.float64)

    h_f, w_f = 1024, 1024
    full_warp_matrix = np.array(
        [
            [linear[0, 0] / w_e, linear[0, 1] / h_e, known["tx"]],
            [linear[1, 0] / w_e, linear[1, 1] / h_e, known["ty"]],
        ],
        dtype=np.float32,
    )
    fundus_vessel_prob = cv2.warpAffine(
        enface_vessel, full_warp_matrix, (w_f, h_f),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    gt_matrix = compose_similarity(known["tx"], known["ty"], cos_t, sin_t, known["scale"], reflect=True)
    result = match_vessels(fundus_vessel_prob, enface_vessel,
                            scale_min=120.0, scale_max=200.0, scale_step=10.0, angle_step_deg=10.0)
    pred_matrix = compose_similarity(
        result["tx"], result["ty"], result["cos_theta"], result["sin_theta"], result["scale"], reflect=True
    )
    err = float(corner_error(pred_matrix, gt_matrix))
    print(f"  conocido:   tx={known['tx']:.2f} ty={known['ty']:.2f} scale={known['scale']:.2f} angle={known['angle_deg']:.1f}")
    print(f"  recuperado: tx={result['tx']:.2f} ty={result['ty']:.2f} scale={result['scale']:.2f} match_score={result['match_score']:.4f}")
    print(f"  corner_error={err:.4f}px")
    ok = err < 2.0
    print("  PASA" if ok else "  FALLA", "- cadena interna autoconsistente" if ok else "- bug real en vessel_template_match.py")
    return ok


def run_oracle_alignment_check() -> None:
    """Parte 2: ver docstring del módulo. Solo diagnóstico, no assert."""
    print("\n=== 2. Oráculo de alineación real (Mock Test) ===")
    cases = find_task2_cases(MOCK_TEST_ROOT)
    for case in cases:
        scenario_dir = case["scenario_dir"]
        frame_id = case["frame_id"]
        vessel_mask_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
        fundus_vessel_gt = (load_label_map(vessel_mask_path) > 0).astype(np.float32)
        volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
        seg_volume = load_volume_label_maps(volume_dir / "Segmentation")
        enface_vessel = enface_vessel_density(seg_volume)
        if enface_vessel.std() < 1e-6:
            print(f"  [{case['scenario']}/{frame_id}] saltado (enface_vessel degenerado, std=0 -> "
                  f"causa raíz del crash reportado: sin clase ArteriesOrVeins en ese volumen)")
            continue

        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)
        params = decompose_similarity(gt_matrix, reflect=True)
        scale = float(params["scale"])
        angle_rad = float(np.arctan2(params["sin_theta"], params["cos_theta"]))

        patch, _ = _build_patch(enface_vessel, scale, angle_rad)
        ph, pw = patch.shape
        tx, ty = float(params["tx"]), float(params["ty"])
        h_f, w_f = fundus_vessel_gt.shape
        x0, y0 = int(round(tx)), int(round(ty))
        region = np.zeros((ph, pw), dtype=np.float32)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + pw, w_f), min(y0 + ph, h_f)
        if sx1 > sx0 and sy1 > sy0:
            region[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = fundus_vessel_gt[sy0:sy1, sx0:sx1]

        ncc = float("nan")
        if patch.std() > 1e-9 and region.std() > 1e-9:
            ncc = float(np.corrcoef(patch.ravel(), region.ravel())[0, 1])
        print(f"  [{case['scenario']}/{frame_id}] scale_GT={scale:.1f} patch_nonzero_frac={(patch > 0).mean():.4f} "
              f"region_nonzero_frac={(region > 0).mean():.4f} oracle_NCC={ncc:.4f}")
    print("  (oracle_NCC ~0 => la señal de enface_vessel_density no correlaciona con el "
          "fundus ni con la posición GT correcta; NO es un bug de búsqueda/geometría)")


def main() -> None:
    ok = run_synthetic_self_consistency_check()
    run_oracle_alignment_check()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
