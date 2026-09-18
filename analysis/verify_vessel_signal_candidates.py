#!/usr/bin/env python3
"""T2-R4 — búsqueda de una señal en-face de vasos que SÍ correlacione con el
fundus, tras el hallazgo de `verify_vessel_template_match_synthetic.py`
(oracle_NCC ~0 para `enface_vessel_density` = fracción de profundidad).

Reutiliza el MISMO oráculo (parámetros REALES del GT vía
`decompose_similarity`, patch alineado en esa posición/escala/ángulo exacta,
NCC contra `arteriesorveins.png` del fundus, `mask>0`) sobre los 5 casos del
Mock Test — no se inventa un oráculo nuevo, se extiende
`run_oracle_alignment_check` para poder variar la SEÑAL de entrada (antes
solo probaba `enface_vessel_density`) sin duplicar la lógica de alineación.

Candidatas probadas, en el orden pedido en la tarea:
  A. `enface_vessel_density` actual (fracción por columna) — referencia, ya
     medida en `verify_vessel_template_match_synthetic.py` (oracle_NCC~0).
  B. Binarizada: `(seg==3).any(axis=1)` — presencia/ausencia por columna.
  C. MIP (max) sobre la MISMA densidad ya calculada (equivalente a B para una
     máscara binaria por vóxel: max de 0/1 == any). Se incluye igual por
     completitud conceptual, pero para `seg==3` (binario por vóxel) MIP y
     `any` son matemáticamente idénticos — no aporta información nueva sobre
     B, se documenta y se salta la ejecución redundante.
  D. Binarizada + dilatación morfológica, kernel 2/3/4 px (motivar tamaño con
     evidencia, no adivinar).

100% CPU (numpy + scipy + opencv), Mock Test local. Correr con:

    python analysis/verify_vessel_signal_candidates.py

Extensión (2026-08-17): ninguna candidata A-D superó el umbral orientativo
(oracle_NCC medio en [0.002, 0.007], vs. umbral 0.3). Dos verificaciones
adicionales antes de declarar el hallazgo, para descartar que el problema
fuera "casi alineado pero desplazado unos píxeles" en vez de "sin señal":

  E. **Dilatación SIMÉTRICA** (ambos lados: patch Y máscara GT del fundus,
     no solo el patch como en D) con kernel 0/1/3/5/7/9/13/17/21 px, sobre
     el patch obtenido con warp directo de la matriz GT completa (sin pasar
     por `_build_patch`/bbox, para descartar un bug de offset ahí también
     — ver función `oracle_ncc_full_warp`). Resultado: NCC medio se mantiene
     en [0.001, 0.010] incluso con dilatación de 21px por lado — no hay
     mejora sustancial en ningún kernel.
  F. **Búsqueda local alrededor del GT** (±30px de escala, ±15° de ángulo,
     dilatación k=5, PERO con traslación libre vía `matchTemplate`, igual
     que hace `match_vessels` de verdad): sube a NCC~0.24-0.31, pero la
     posición ganadora NO coincide con la posición real del GT (verificado
     comparando `max_loc` contra las esquinas GT) — es un falso positivo por
     autosimilitud del árbol vascular (patrones de curvas finas se parecen
     entre sí en cualquier parte de la imagen), exactamente el modo de fallo
     que ya producía `corner_error=573px` en T2-R4/Motor 1. Confirma que el
     problema no es "small misalignment", es ausencia de señal distintiva.

Conclusión: ninguna transformación de `enface_vessel_density` (binarizar,
proyectar distinto, dilatar uno o ambos lados) resuelve el problema. La
proyección en-face de la clase ArteriesOrVeins del volumen OCT, aun en la
posición/escala/ángulo EXACTOS del GT, no reproduce el árbol vascular visible
en el fundus con fidelidad suficiente para template matching clásico.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fido.data.common import enface_vessel_density, load_label_map, load_volume_label_maps  # noqa: E402
from fido.data.task2 import find_task2_cases  # noqa: E402
from fido.geometry import decompose_similarity  # noqa: E402
from fido.models.vessel_template_match import _build_patch  # noqa: E402

MOCK_TEST_ROOT = PROJECT_ROOT / "data" / "Mock Test" / "Task 2"


def sig_density(seg_volume: np.ndarray) -> np.ndarray:
    """A. Referencia actual — fracción de profundidad ocupada por el vaso."""
    return enface_vessel_density(seg_volume)


def sig_binary(seg_volume: np.ndarray) -> np.ndarray:
    """B. Presencia/ausencia de vaso en cualquier profundidad de la columna."""
    return (seg_volume == 3).any(axis=1).astype(np.float32)


def sig_binary_dilated(seg_volume: np.ndarray, kernel_px: int) -> np.ndarray:
    """D. B + dilatación morfológica cuadrada de kernel_px x kernel_px."""
    binary = (seg_volume == 3).any(axis=1)
    structure = np.ones((kernel_px, kernel_px), dtype=bool)
    return binary_dilation(binary, structure=structure).astype(np.float32)


def oracle_ncc_for_case(case: dict, signal_fn) -> tuple[float, dict]:
    """MISMA lógica que `run_oracle_alignment_check` en
    verify_vessel_template_match_synthetic.py, parametrizada por la función de
    señal. Devuelve (ncc, info_diagnostico)."""
    scenario_dir = case["scenario_dir"]
    frame_id = case["frame_id"]
    vessel_mask_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
    fundus_vessel_gt = (load_label_map(vessel_mask_path) > 0).astype(np.float32)
    volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
    seg_volume = load_volume_label_maps(volume_dir / "Segmentation")

    enface_vessel = signal_fn(seg_volume)
    if enface_vessel.std() < 1e-6:
        return float("nan"), {"skipped": True}

    data = json.loads(case["json_path"].read_text(encoding="utf-8"))
    gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)
    params = decompose_similarity(gt_matrix, reflect=True)
    scale = float(params["scale"])
    angle_rad = float(np.arctan2(params["sin_theta"], params["cos_theta"]))

    patch, corners_patch_local = _build_patch(enface_vessel, scale, angle_rad)
    ph, pw = patch.shape
    tx, ty = float(params["tx"]), float(params["ty"])
    h_f, w_f = fundus_vessel_gt.shape
    # CORRECCIÓN respecto al oráculo original: (tx,ty) es la posición en el
    # fundus de la esquina uv=(0,0) del cuadrado unitario (por construcción
    # de compose_similarity: M@[0,0,1]=[tx,ty]), NO la esquina superior
    # izquierda del bounding box del patch — esas coinciden solo cuando
    # bbox_min=(0,0), que no es el caso general (rotación != 0/90/180/270°).
    # `corners_patch_local[0]` es la posición LOCAL de esa misma esquina
    # dentro del patch (por construcción de `_build_patch`), así que la
    # esquina superior-izquierda real del patch en el fundus es
    # (tx,ty) - corners_patch_local[0]. Verificado con un warp directo de la
    # matriz GT completa (sin pasar por `_build_patch`) como método
    # independiente — ambos coinciden (ver `oracle_ncc_full_warp`), y ambos
    # dan el mismo resultado cualitativo (NCC~0) que el oráculo original con
    # el bug, así que la corrección NO cambia la conclusión, pero es la
    # versión correcta y es la que se usa de aquí en adelante.
    corner00_local = corners_patch_local[0]
    x0 = int(round(tx - corner00_local[0]))
    y0 = int(round(ty - corner00_local[1]))
    region = np.zeros((ph, pw), dtype=np.float32)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + pw, w_f), min(y0 + ph, h_f)
    if sx1 > sx0 and sy1 > sy0:
        region[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = fundus_vessel_gt[sy0:sy1, sx0:sx1]

    ncc = float("nan")
    if patch.std() > 1e-9 and region.std() > 1e-9:
        ncc = float(np.corrcoef(patch.ravel(), region.ravel())[0, 1])
    info = {
        "skipped": False,
        "patch_nonzero_frac": float((patch > 0).mean()),
        "region_nonzero_frac": float((region > 0).mean()),
    }
    return ncc, info


def oracle_ncc_full_warp(case: dict, signal_fn, dilate_k: int = 0) -> float:
    """E. Verificación independiente que NO pasa por `_build_patch`/bbox: usa
    la matriz GT completa (`a,b,tx,c,d,ty`) directamente para construir un
    único `cv2.warpAffine` de la señal en-face completa sobre el lienzo del
    fundus (misma fórmula que `verify_vessel_template_match_synthetic.py`
    usa para el caso sintético), y compara contra la máscara GT del fundus
    — con dilatación SIMÉTRICA opcional (`dilate_k`, aplicada a AMBOS lados)
    para tolerar desalineaciones de unos pocos píxeles sin premiar solo el
    solapamiento exacto pixel a pixel."""
    scenario_dir = case["scenario_dir"]
    frame_id = case["frame_id"]
    vessel_mask_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
    fundus_vessel_gt = (load_label_map(vessel_mask_path) > 0).astype(np.float32)
    volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
    seg_volume = load_volume_label_maps(volume_dir / "Segmentation")
    enface_vessel = signal_fn(seg_volume)
    if enface_vessel.std() < 1e-6:
        return float("nan")

    data = json.loads(case["json_path"].read_text(encoding="utf-8"))
    gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)
    a, b, tx = gt_matrix[0]
    c, d, ty = gt_matrix[1]
    h_e, w_e = enface_vessel.shape
    h_f, w_f = fundus_vessel_gt.shape
    full_warp = np.array([[a / w_e, b / h_e, tx], [c / w_e, d / h_e, ty]], dtype=np.float32)
    warped = cv2.warpAffine(
        enface_vessel.astype(np.float32), full_warp, (w_f, h_f),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    gt = fundus_vessel_gt
    if dilate_k > 0:
        kernel = np.ones((dilate_k, dilate_k), np.float32)
        warped = cv2.dilate(warped, kernel)
        gt = cv2.dilate(gt, kernel)
    if warped.std() > 1e-9 and gt.std() > 1e-9:
        return float(np.corrcoef(warped.ravel(), gt.ravel())[0, 1])
    return float("nan")


def local_search_diagnostic(case: dict) -> None:
    """F. Diagnóstico (no forma parte de la decisión de la candidata
    ganadora): búsqueda LOCAL de escala/ángulo alrededor del GT (±30px,
    ±15°) con dilatación k=5 en ambos lados, pero traslación LIBRE vía
    `cv2.matchTemplate` (igual que hace `match_vessels` de verdad). Si el
    mejor match encontrado NO cae cerca de la posición real del GT, confirma
    que el patrón es autosimilar (cualquier parche de vasos finos se parece
    a cualquier otro) y no que "casi" se alinea con un pequeño error de
    traslación."""
    scenario_dir = case["scenario_dir"]
    frame_id = case["frame_id"]
    vessel_mask_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
    fundus_vessel_gt = (load_label_map(vessel_mask_path) > 0).astype(np.float32)
    volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
    seg_volume = load_volume_label_maps(volume_dir / "Segmentation")
    enface_binary = sig_binary(seg_volume)
    if enface_binary.std() < 1e-6:
        print(f"  [{case['scenario']}/{frame_id}] saltado (señal degenerada)")
        return

    data = json.loads(case["json_path"].read_text(encoding="utf-8"))
    gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)
    params = decompose_similarity(gt_matrix, reflect=True)
    scale_gt = float(params["scale"])
    angle_gt = float(np.arctan2(params["sin_theta"], params["cos_theta"]))

    best_score, best_info = -2.0, None
    kernel5 = np.ones((5, 5), np.float32)
    for scale in np.arange(scale_gt - 30, scale_gt + 30 + 1e-6, 5.0):
        for angle_off_deg in np.arange(-15, 15 + 1e-6, 3.0):
            angle_rad = angle_gt + np.radians(angle_off_deg)
            patch, _ = _build_patch(enface_binary, scale, angle_rad)
            if patch.std() < 1e-6:
                continue
            patch_d = cv2.dilate(patch, kernel5)
            fundus_d = cv2.dilate(fundus_vessel_gt, kernel5)
            result = cv2.matchTemplate(fundus_d, patch_d, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score, best_info = max_val, (scale, np.degrees(angle_rad), max_loc)

    print(
        f"  [{case['scenario']}/{frame_id}] mejor_NCC_local={best_score:.4f}  "
        f"GT: tx={params['tx']:.0f} ty={params['ty']:.0f} scale={scale_gt:.0f}  "
        f"encontrado: scale={best_info[0]:.0f} angle={best_info[1]:.0f} loc={best_info[2]}"
    )


def main() -> None:
    cases = find_task2_cases(MOCK_TEST_ROOT)
    if not cases:
        raise SystemExit(f"No se encontraron casos en {MOCK_TEST_ROOT}")

    candidates = [
        ("A. density (actual, referencia)", sig_density),
        ("B. binary any(axis=1)", sig_binary),
        ("D. binary + dilate k=2", lambda sv: sig_binary_dilated(sv, 2)),
        ("D. binary + dilate k=3", lambda sv: sig_binary_dilated(sv, 3)),
        ("D. binary + dilate k=4", lambda sv: sig_binary_dilated(sv, 4)),
        ("D. binary + dilate k=6", lambda sv: sig_binary_dilated(sv, 6)),
        ("D. binary + dilate k=8", lambda sv: sig_binary_dilated(sv, 8)),
    ]

    print(f"{len(cases)} casos encontrados en {MOCK_TEST_ROOT}\n")
    print("NOTA: C. (MIP/max) es matemáticamente idéntica a B. para una máscara")
    print("binaria por vóxel (max(0/1)==any) — se omite la corrida redundante.\n")

    summary = {}
    for name, fn in candidates:
        print(f"=== {name} ===")
        nccs = []
        for case in cases:
            ncc, info = oracle_ncc_for_case(case, fn)
            if info.get("skipped"):
                print(f"  [{case['scenario']}/{case['frame_id']}] saltado (señal degenerada, std=0)")
                continue
            nccs.append(ncc)
            print(
                f"  [{case['scenario']}/{case['frame_id']}] oracle_NCC={ncc:.4f}  "
                f"patch_nonzero={info['patch_nonzero_frac']:.4f}  region_nonzero={info['region_nonzero_frac']:.4f}"
            )
        arr = np.array(nccs, dtype=np.float64)
        mean_ncc = float(np.nanmean(arr)) if len(arr) else float("nan")
        summary[name] = mean_ncc
        print(f"  media oracle_NCC = {mean_ncc:.4f}  (n={len(arr)})\n")

    print("=== Resumen A-D (oráculo GT exacto) ===")
    for name, mean_ncc in summary.items():
        flag = " <-- umbral orientativo 0.3 superado" if mean_ncc > 0.3 else ""
        print(f"  {name:35s} media oracle_NCC = {mean_ncc:.4f}{flag}")

    print("\n=== E. Dilatación SIMÉTRICA (patch Y fundus GT), warp directo de matriz GT completa ===")
    for k in (0, 1, 3, 5, 7, 9, 13, 17, 21):
        nccs = [oracle_ncc_full_warp(case, sig_binary, dilate_k=k) for case in cases]
        arr = np.array(nccs, dtype=np.float64)
        print(f"  dilate_k={k:3d}  media oracle_NCC = {np.nanmean(arr):.4f}  por_caso={np.round(arr, 4)}")

    print("\n=== F. Diagnóstico: búsqueda local (traslación libre) alrededor del GT ===")
    for case in cases:
        local_search_diagnostic(case)
    print(
        "  (si la posición ganadora no coincide con la del GT => autosimilitud del "
        "árbol vascular, no 'casi alineado')"
    )


if __name__ == "__main__":
    main()
