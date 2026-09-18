"""Geometría compartida de Task 2: parametrización de 4 DOF, proyección de
esquinas, y la métrica AUC exacta que usa el scoring oficial.

Agnóstico a arquitectura — lo usan por igual un baseline de heatmap, DPCN++, o
cualquier método de la escalera. Ver ATTACK_LADDER.md sección Task 2.

La convención de coordenadas replica exactamente
`vendor/fido/Codabench Bundle/scoring_program/scoring_registration.py`:
las 4 esquinas del cuadrado unitario canónico se proyectan por la matriz
predicha y por la real, y el score es el AUC del error medio de esquinas.
"""

from __future__ import annotations

import numpy as np
import torch

# Las mismas 4 esquinas homogéneas que usa scoring_registration.py.
UNIT_SQUARE_CORNERS = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    dtype=np.float64,
)

MAX_THRESHOLD_PX = 10  # idéntico a MAX_THRESHOLD_PX en el bundle oficial


def compose_similarity(tx, ty, cos_theta, sin_theta, scale, reflect: bool = True):
    """Arma la matriz 3x3 desde la parametrización de 4 DOF.

    M = [[ s*cosθ,  s*sinθ, tx],
         [ s*sinθ, -s*cosθ, ty],   <- signos de reflexión: medidos en el
         [      0,       0,  1]]     100% de los 1214 casos de entrenamiento

    `reflect=False` da la similitud propia (rotación estándar, sin reflexión):
    [[s*cosθ, -s*sinθ, tx], [s*sinθ, s*cosθ, ty]] — matriz ortogonal con
    det=+s² (a diferencia de la de arriba, det=-s²). Útil para verificar la
    hipótesis de reflexión fija contra datos frescos (T2-R1) o para componer
    matrices que SÍ deben ser una rotación propia (p.ej. `fit_closed_form_similarity`,
    que recupera una rotación sin reflexión vía Kabsch/SVD).

    Acepta tensores de PyTorch (batched, última dim = parámetros) o floats.
    Devuelve el mismo tipo que la entrada.
    """
    is_tensor = isinstance(tx, torch.Tensor)
    stack = torch.stack if is_tensor else np.stack

    a = scale * cos_theta
    b = (scale * sin_theta) if reflect else (-scale * sin_theta)
    c = scale * sin_theta
    d = (-scale * cos_theta) if reflect else (scale * cos_theta)
    zero = torch.zeros_like(tx) if is_tensor else np.zeros_like(np.asarray(tx))
    one = torch.ones_like(tx) if is_tensor else np.ones_like(np.asarray(tx))

    row0 = stack([a, b, tx], axis=-1)
    row1 = stack([c, d, ty], axis=-1)
    row2 = stack([zero, zero, one], axis=-1)
    return stack([row0, row1, row2], axis=-2)


def decompose_similarity(matrix, reflect: bool = True):
    """Inversa de `compose_similarity`: recupera (tx,ty,cosθ,sinθ,scale) desde
    una matriz 3x3 ya conocida (p.ej. el GT de Task 2 leído del JSON).

    Solo usa la fila 0 (`a=s·cosθ`, `b=±s·sinθ` según `reflect`, `tx`) y la
    traslación de la fila 1 (`ty`) — no depende de `d` para recuperar θ/s.
    `reflect` debe coincidir con el que se usó (o se usará) en
    `compose_similarity`: para el GT real de Task 2 siempre es `True` (T2-R1/R3,
    100% de los 1214 casos verificados), que es el default.

    Acepta tensores de PyTorch (..., 3, 3) o arrays/floats de NumPy. Devuelve
    un dict con las mismas claves que espera `compose_similarity(**dict)`.
    """
    is_tensor = isinstance(matrix, torch.Tensor)
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    tx = matrix[..., 0, 2]
    ty = matrix[..., 1, 2]

    scale = torch.sqrt(a**2 + b**2) if is_tensor else np.sqrt(a**2 + b**2)
    cos_theta = a / scale
    sin_theta = (b / scale) if reflect else (-b / scale)
    return dict(tx=tx, ty=ty, cos_theta=cos_theta, sin_theta=sin_theta, scale=scale)


def project_corners(matrix):
    """Proyecta las 4 esquinas del cuadrado unitario por `matrix`.

    `matrix`: (..., 3, 3). Devuelve (..., 4, 2).
    Réplica exacta de `project_corners` en scoring_registration.py.
    """
    is_tensor = isinstance(matrix, torch.Tensor)
    if is_tensor:
        corners_h = torch.as_tensor(UNIT_SQUARE_CORNERS, dtype=matrix.dtype, device=matrix.device)
        projected = torch.einsum("...ij,kj->...ki", matrix, corners_h)
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        projected = np.einsum("...ij,kj->...ki", matrix, UNIT_SQUARE_CORNERS)
    return projected[..., :2] / projected[..., 2:3]


def corner_error(pred_matrix, gt_matrix):
    """Error medio de esquinas entre dos matrices — la cantidad que puntúa
    el challenge. Soporta batch. Devuelve (...,) — un escalar por muestra."""
    pred_corners = project_corners(pred_matrix)
    gt_corners = project_corners(gt_matrix)
    diff = pred_corners - gt_corners
    if isinstance(diff, torch.Tensor):
        return torch.linalg.norm(diff, dim=-1).mean(dim=-1)
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def corner_auc(errors, max_threshold: int = MAX_THRESHOLD_PX) -> float:
    """AUC de la fracción de casos con error <= t, promediada sobre
    t = 0..max_threshold enteros. Réplica exacta de `auc_from_errors` del
    scoring oficial. `errors` es un array 1D de errores de esquina en px."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("Cannot compute AUC with no errors")
    total = 0.0
    for threshold in range(max_threshold + 1):
        total += float(np.mean(errors <= threshold))
    return total / (max_threshold + 1)


def huber_saturated(errors, delta: float = 5.0, saturation: float = 15.0):
    """Pérdida robusta para entrenar sobre el error de esquinas.

    Justificación (literature/01_task2_registration.md §5, brecha #6): la AUC
    con umbrales 0..10 hace que un error de 200px valga exactamente lo mismo
    que uno de 50px — ambos pierden los 11 umbrales. Una L2 estándar gasta
    gradiente en reducir errores ya irrecuperables. Esta pérdida es Huber
    hasta `saturation` px y CONSTANTE después: no hay incentivo de gradiente
    en empujar un fallo catastrófico a "menos catastrófico".
    """
    is_tensor = isinstance(errors, torch.Tensor)
    abs_err = torch.abs(errors) if is_tensor else np.abs(errors)
    clipped = torch.clamp(abs_err, max=saturation) if is_tensor else np.clip(abs_err, None, saturation)
    quadratic = 0.5 * clipped**2
    linear = delta * (clipped - 0.5 * delta)
    if is_tensor:
        return torch.where(clipped <= delta, quadratic, linear)
    return np.where(clipped <= delta, quadratic, linear)


def fit_closed_form_similarity(pred_corners, ref_corners=UNIT_SQUARE_CORNERS[:, :2]):
    """Recupera (tx,ty,cosθ,sinθ,s) que mejor mapea `ref_corners` a
    `pred_corners` en el sentido de mínimos cuadrados (similitud propia).

    Útil para T2-R2/T2-R5: si un modelo predice directamente las 4 esquinas
    (parametrización de DeTone), esto reconstruye la matriz en forma cerrada
    sin resolver un sistema lineal general — es la solución de Procrustes
    ortogonal restringida a rotación+escala+traslación (sin reflexión; el
    signo se aplica después con `compose_similarity(..., reflect=True)`
    reconstruyendo desde los parámetros recuperados aquí).
    """
    ref = np.asarray(ref_corners, dtype=np.float64)
    pred = np.asarray(pred_corners, dtype=np.float64)

    ref_mean = ref.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    ref_centered = ref - ref_mean
    pred_centered = pred - pred_mean

    # Umeyama/Kabsch 2D restringido a rotación (sin reflexión) + escala
    # uniforme: minimiza ||s*R*ref_centered + t - pred_centered||^2.
    covariance = pred_centered.T @ ref_centered / len(ref)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    variance_ref = (ref_centered**2).sum() / len(ref)
    scale = singular_values.sum() / variance_ref if variance_ref > 0 else 1.0
    translation = pred_mean - scale * rotation @ ref_mean

    cos_theta, sin_theta = rotation[0, 0], rotation[1, 0]
    return dict(tx=translation[0], ty=translation[1],
               cos_theta=cos_theta, sin_theta=sin_theta, scale=scale)
