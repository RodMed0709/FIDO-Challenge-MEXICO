"""Motor 1 de T2-R4: template matching clásico sobre el patrón de vasos —
literatura de referencia: RetinaMatch >94% éxito, Noyel 96% sobre 271 pares,
ambos con estructura vascular como señal (ver ATTACK_LADDER.md, T2-R4).

Decisión de diseño (correlación de fase vs NCC): se usa `cv2.matchTemplate`
con `TM_CCOEFF_NORMED` (normalized cross-correlation, no `cv2.phaseCorrelate`)
porque `phaseCorrelate` solo resuelve TRASLACIÓN pura entre dos imágenes del
MISMO tamaño — no da directamente escala ni rotación, y asume que ambas
imágenes comparten el mismo campo de visión (razonable para registrar dos
fotos casi idénticas, no para un template pequeño de vasos dentro de un
fundus 1024×1024 donde el área común es ~2.4% del total, ver T2-R5). NCC vía
`matchTemplate` sí resuelve traslación de un template MÁS CHICO dentro de una
imagen más grande directamente, que es exactamente la estructura del
problema aquí. Rotación y escala se resuelven por búsqueda en grilla (grid
search) sobre los priors ya medidos en T2-R3/T2-R8 (escala ~160±13.7px,
ángulo uniforme en [0,2π)) — no por invarianza del método, por diseño
explícito: no se justifica log-polar (T2-R5, correlación de fase
diferenciable) todavía, es el peldaño siguiente si esto no alcanza.

Manejo de la reflexión (determinante negativo, 100% de los casos verificado
en T2-R1/T2-R3): la matriz lineal de `compose_similarity(reflect=True)` es

    A = s * [[cosθ, sinθ], [sinθ, -cosθ]] = s * R(θ) @ diag(1, -1)

(R(θ) = matriz de rotación propia estándar). Se construye A directamente con
esta fórmula (reflect=True SIEMPRE, no se busca "con o sin reflexión" — ya
está resuelto por T2-R1/T2-R3, buscar esa rama sería desperdiciar mitad de
la grilla en una hipótesis descartada). El patrón de vasos rotado+escalado
+reflejado se renderiza directamente desde `enface_vessel` vía
`cv2.warpAffine` con esta A (no se pre-flipea la imagen a mano ni se
reordenan esquinas "a ojo" — eso es frágil y difícil de verificar; construir
el patch con la fórmula exacta de `compose_similarity` y trackear la
posición de las 4 esquinas del cuadrado unitario A TRAVÉS del mismo warp es
inequívoco).

Recuperación de parámetros: NO se resuelve manualmente el ajuste — se arma
`ref_corners_reflected = diag(1,-1) @ UNIT_SQUARE_CORNERS` (aplicar la
reflexión al cuadrado unitario canónico) y se llama a
`fit_closed_form_similarity(pred_corners, ref_corners=ref_corners_reflected)`
(la Kabsch/Procrustes YA VERIFICADA en `geometry.py`, no reimplementada).
Por construcción algebraica (ver derivación arriba), la relación entre
`ref_corners_reflected` y las 4 esquinas proyectadas en el fundus
(`pred_corners`) es una similitud PROPIA exacta (sin reflexión adicional
dentro del ajuste) — exactamente el caso que `fit_closed_form_similarity`
sabe resolver — y los parámetros que devuelve son directamente los que
espera `compose_similarity(..., reflect=True)`, sin ningún ajuste de signo
posterior.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependencia verificada al inicio de T2-R4
    raise ImportError(
        "vessel_template_match requiere opencv-python (o headless). "
        "Instalar con: pip install opencv-python-headless"
    ) from exc

from fido.geometry import fit_closed_form_similarity

# Cuadrado unitario canónico ya reflejado (diag(1,-1) aplicado a
# UNIT_SQUARE_CORNERS de geometry.py) — ES el `ref_corners` correcto para
# recuperar los parámetros de `compose_similarity(reflect=True)` vía
# `fit_closed_form_similarity` (ver derivación en el docstring del módulo).
_REF_CORNERS_REFLECTED = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [0.0, -1.0]], dtype=np.float64
)
_UNIT_SQUARE_UV = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
)


def _build_patch(enface_vessel: np.ndarray, scale: float, angle_rad: float):
    """Renderiza el patch rotado+escalado+reflejado del patrón de vasos
    en-face, y devuelve junto con él la posición (en coordenadas LOCALES del
    patch) de las 4 esquinas del cuadrado unitario, en el mismo orden que
    `UNIT_SQUARE_CORNERS`. Ninguna traslación global (`tx`,`ty`) entra
    todavía aquí — eso lo resuelve `cv2.matchTemplate` después."""
    h_e, w_e = enface_vessel.shape
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
    # Misma fórmula EXACTA que compose_similarity(reflect=True) en geometry.py.
    linear = scale * np.array([[cos_t, sin_t], [sin_t, -cos_t]], dtype=np.float64)

    corners_local = _UNIT_SQUARE_UV @ linear.T  # (4,2), sin trasladar aún
    bbox_min = corners_local.min(axis=0)
    bbox_max = corners_local.max(axis=0)
    patch_w = int(np.ceil(bbox_max[0] - bbox_min[0])) + 1
    patch_h = int(np.ceil(bbox_max[1] - bbox_min[1])) + 1

    # Mapea píxel (col,row) del en-face -> (u,v)=(col/w_e,row/h_e) -> `linear`
    # -> coordenada LOCAL del patch (restando bbox_min para que el patch
    # empiece en (0,0)).
    warp_matrix = np.array(
        [
            [linear[0, 0] / w_e, linear[0, 1] / h_e, -bbox_min[0]],
            [linear[1, 0] / w_e, linear[1, 1] / h_e, -bbox_min[1]],
        ],
        dtype=np.float32,
    )
    patch = cv2.warpAffine(
        enface_vessel.astype(np.float32), warp_matrix, (patch_w, patch_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    corners_patch_local = corners_local - bbox_min  # (4,2), coords dentro del patch
    return patch, corners_patch_local


def match_vessels(
    fundus_vessel_prob: np.ndarray,
    enface_vessel: np.ndarray,
    scale_min: float = 120.0,
    scale_max: float = 200.0,
    scale_step: float = 10.0,
    angle_step_deg: float = 10.0,
) -> dict:
    """Busca la posición/escala/ángulo que mejor alinea el patrón de vasos
    de `enface_vessel` (densidad de vaso del volumen OCT, (Ht,Wt)) dentro de
    `fundus_vessel_prob` (mapa de probabilidad/máscara de vaso del fundus,
    (Hf,Wf)) por búsqueda en grilla (escala, ángulo) + NCC (`matchTemplate`)
    para la traslación en cada celda de la grilla.

    Rango de búsqueda por defecto centrado en los priors ya medidos sobre
    1214 casos reales de entrenamiento (T2-R3/ATTACK_LADDER: escala
    160±13.7px, ángulo uniforme en [0,2π)) — `scale_min`/`scale_max` cubren
    ~holgadamente 120-200px (>3 desviaciones estándar), `angle_step_deg`
    recorre el círculo completo ya que no hay prior angular.

    Devuelve un dict `{tx, ty, cos_theta, sin_theta, scale, match_score}` —
    mismo formato de parámetros que `compose_similarity`/`decompose_similarity`
    de `geometry.py` (con `reflect=True`, la convención verificada del GT
    real de Task 2). `match_score` es la correlación normalizada (NCC) de la
    mejor celda, en [-1,1] — no es parte del contrato de `compose_similarity`,
    es diagnóstico extra para decidir si el match es confiable.
    """
    fundus = fundus_vessel_prob.astype(np.float32)
    h_f, w_f = fundus.shape

    best = None
    scales = np.arange(scale_min, scale_max + 1e-6, scale_step)
    angles_deg = np.arange(0.0, 360.0, angle_step_deg)

    for scale in scales:
        for angle_deg in angles_deg:
            angle_rad = np.radians(angle_deg)
            patch, corners_patch_local = _build_patch(enface_vessel, float(scale), angle_rad)
            patch_h, patch_w = patch.shape
            if patch_h < 4 or patch_w < 4 or patch_h > h_f or patch_w > w_f:
                continue
            if patch.std() < 1e-6:
                # Patch degenerado (p.ej. sin señal de vaso en esa orientación
                # tras el warp) -> NCC indefinida, se descarta la celda.
                continue

            result = cv2.matchTemplate(fundus, patch, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if best is None or max_val > best["score"]:
                window_xy = np.array(max_loc, dtype=np.float64)  # (x,y) top-left del patch en fundus
                pred_corners = corners_patch_local + window_xy  # (4,2), MISMO orden que UNIT_SQUARE_CORNERS
                best = {
                    "score": float(max_val),
                    "scale": float(scale),
                    "angle_deg": float(angle_deg),
                    "pred_corners": pred_corners,
                }

    if best is None:
        raise RuntimeError(
            "match_vessels: ninguna celda de la grilla produjo un patch válido "
            "(revisar tamaño de fundus_vessel_prob/enface_vessel o el rango de escalas)."
        )

    params = fit_closed_form_similarity(best["pred_corners"], ref_corners=_REF_CORNERS_REFLECTED)
    return {
        "tx": float(params["tx"]),
        "ty": float(params["ty"]),
        "cos_theta": float(params["cos_theta"]),
        "sin_theta": float(params["sin_theta"]),
        "scale": float(params["scale"]),
        "match_score": best["score"],
    }
