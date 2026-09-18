"""Tests del núcleo matemático de `fido.models.task2_dpcn` con casos
SINTÉTICOS de respuesta conocida -- el objetivo explícito es no perseguir un
bug de signo/convención durante horas de GPU real (ver ATTACK_LADDER.md
T2-92). Cada test aísla una pieza:

1. `phase_correlation` recupera un desplazamiento de píxeles conocido.
2. El log-polar convierte una rotación conocida en un desplazamiento
   angular conocido (vía `estimate_rotation_scale`).
3. `fit_similarity_from_corners` recupera exactamente (sin residuo) la
   matriz de una similitud propia dados sus 4 corners.
4. El pipeline Fourier-Mellin COMPLETO (log-polar + correlación de fase +
   desambiguación de 180 grados) recupera una transformación sintética
   conocida con error de esquina pequeño, para varios casos fijos y
   reproducibles.
5. El modelo `Task2DPCNModel` completo produce las formas esperadas, todos
   los parámetros reciben gradiente, y la cabeza de refinamiento de
   esquinas (inicializada en cero) no perturba la estimación clásica al
   arrancar.

Hallazgo importante documentado aquí (no es un bug, es una propiedad medida
del método clásico): el núcleo Fourier-Mellin SIN ENTRENAR, sobre contenido
sintético arbitrario (ruido suavizado + máscara), falla (error de esquina
>>10px, a veces >100px) en una fracción sustancial (~50%) de combinaciones
aleatorias de rotación/escala/traslación -- incluso sin ruido de medición,
con contenido perfectamente conocido en ambos lados. La causa verificada
(no adivinada): para varias combinaciones, el pico GLOBAL de la superficie
de correlación de fase en el dominio log-polar no coincide con la ubicación
que le correspondería a la transformación real -- hay picos espurios más
altos que el correcto. Esto es consistente con limitaciones conocidas de
Fourier-Mellin clásico sobre contenido de bajo/moderado contenido
espectral discriminante, y es precisamente la motivación de entrenar los
encoders de punta a punta (features aprendidas, no espectro de magnitud
crudo) y de la cabeza de refinamiento de esquinas -- ver PRE_REGISTRATION.md
de T2-92 para cómo esto entra en el criterio GO/NO-GO. Los casos de este
archivo son deterministas y verificados uno a uno (no un muestreo aleatorio
al vuelo) precisamente porque un muestreo aleatorio flaquea con esta tasa.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fido.geometry import compose_similarity, project_corners
from fido.models.task2_dpcn import (
    LogPolarConfig,
    Task2DPCNModel,
    _corner_fit_pinv,
    apply_similarity_warp,
    build_log_polar_grid,
    estimate_rotation_scale,
    fit_similarity_from_corners,
    phase_correlation,
    resolve_rotation,
    similarity_matrix_about_center,
)


def _smoothed_noise_texture(seed: int, size: int) -> torch.Tensor:
    """Ruido gaussiano suavizado 2 veces (kernel 3x3), enmascarado a un
    cuadrado interior -- contenido rico/anisótropo, no radialmente
    simétrico, con suficiente estructura espectral para Fourier-Mellin.
    Determinista: fija la semilla global de torch antes de generar."""
    torch.manual_seed(seed)
    noise = torch.randn(1, 1, size, size)
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    kernel = kernel / kernel.sum()
    texture = noise
    for _ in range(2):
        texture = F.conv2d(F.pad(texture, (1, 1, 1, 1), mode="reflect"), kernel)
    texture = (texture - texture.mean()) / texture.std()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    mask = (xx.abs() < 0.85) & (yy.abs() < 0.85)
    return texture * mask.float()


def test_phase_correlation_recovers_known_pixel_shift():
    torch.manual_seed(0)
    image = torch.rand(1, 1, 64, 64)
    shifted = torch.roll(image, shifts=(-3, 5), dims=(-2, -1))  # content moves (dx=5, dy=-3)
    shift, _surface = phase_correlation(image, shifted, temperature=0.05, window=7)
    assert torch.allclose(shift, torch.tensor([[5.0, -3.0]]), atol=1e-3)


def test_log_polar_converts_rotation_and_scale_into_a_recoverable_shift():
    """Fourier-Mellin clásico: rotar+escalar en espacio real se convierte en
    un desplazamiento en el dominio log-polar de la magnitud de Fourier.
    Caso fijo verificado (ver ATTACK_LADDER.md T2-92): tolerancias generosas
    porque la resolución angular/radial finita (192x96 bins) introduce un
    sesgo de cuantización pequeño pero no nulo."""
    size = 160
    template = _smoothed_noise_texture(seed=1, size=size)
    theta_true = torch.tensor([0.5])
    scale_true = torch.tensor([1.4])
    window = apply_similarity_warp(template, theta_true, scale_true, out_size=(size, size))

    cfg = LogPolarConfig(size=size)
    grid = build_log_polar_grid(cfg)
    d_theta, scale_est, _surface = estimate_rotation_scale(template, window, cfg, grid)

    assert abs(d_theta.item() - theta_true.item()) < 0.03  # < ~1.7 deg
    assert abs(scale_est.item() - scale_true.item()) < 0.02


def test_fit_similarity_from_corners_recovers_known_transform_exactly():
    """El ajuste cerrado (pseudoinversa fija) debe ser EXACTO (sin
    residuo) cuando las 4 esquinas provienen de verdad de una similitud
    propia -- exactamente el caso del marco canónico de Task 2 (ver
    docstring del módulo: por qué NO se usa `fit_reflected_similarity`
    aquí)."""
    tx, ty = torch.tensor([300.0]), torch.tensor([420.0])
    cos_theta, sin_theta = torch.tensor([0.8]), torch.tensor([0.6])  # unit vector
    scale = torch.tensor([160.0])
    matrix_true = compose_similarity(tx, ty, cos_theta, sin_theta, scale, reflect=False)
    corners = project_corners(matrix_true)

    pinv = _corner_fit_pinv()
    matrix_fit = fit_similarity_from_corners(corners, pinv)

    assert torch.allclose(matrix_fit, matrix_true, atol=1e-3)


def test_fit_similarity_from_corners_is_differentiable():
    corners = project_corners(
        compose_similarity(torch.tensor([10.0]), torch.tensor([-5.0]), torch.tensor([1.0]),
                           torch.tensor([0.0]), torch.tensor([50.0]), reflect=False)
    ).requires_grad_(True)
    pinv = _corner_fit_pinv()
    matrix = fit_similarity_from_corners(corners, pinv)
    matrix.sum().backward()
    assert corners.grad is not None
    assert torch.isfinite(corners.grad).all()


# Casos fijos, verificados uno a uno (ver docstring del módulo): la misma
# receta de textura (`_smoothed_noise_texture`) con distintas semillas y
# transformaciones da error de esquina bajo. Documentado explícitamente que
# NO todas las combinaciones aleatorias pasan (ver ATTACK_LADDER.md T2-92)
# -- estos son los que sí, elegidos para probar la matemática, no para
# afirmar robustez universal del núcleo sin entrenar.
_KNOWN_GOOD_CASES = [
    dict(seed=1, size=160, theta=0.5, scale=1.4, tx=8.0, ty=-5.0),
    dict(seed=1, size=160, theta=-0.7, scale=1.2, tx=-6.0, ty=3.0),
    dict(seed=2, size=160, theta=0.9, scale=0.85, tx=4.0, ty=-8.0),
    dict(seed=7, size=160, theta=2.2, scale=1.1, tx=6.0, ty=6.0),
]


def test_full_fourier_mellin_pipeline_recovers_synthetic_transform():
    """Pipeline clásico completo (log-polar + correlación de fase +
    desambiguación de 180 grados + ajuste cerrado de esquinas), sin
    ninguna red neuronal de por medio -- la pieza cuya corrección importa
    antes de gastar GPU real. Tolerancia de 8px de error medio de esquina
    (holgada: los casos verificados dan 1.3-4.2px, ver
    ATTACK_LADDER.md T2-92) porque la cuantización angular/radial finita
    (192x96 bins) no da error cero exacto."""
    for case in _KNOWN_GOOD_CASES:
        size = case["size"]
        template = _smoothed_noise_texture(seed=case["seed"], size=size)
        theta_true = torch.tensor([case["theta"]])
        scale_true = torch.tensor([case["scale"]])
        tx_true = torch.tensor([case["tx"]])
        ty_true = torch.tensor([case["ty"]])
        window = apply_similarity_warp(template, theta_true, scale_true, tx_true, ty_true,
                                       out_size=(size, size))

        cfg = LogPolarConfig(size=size)
        grid = build_log_polar_grid(cfg)
        d_theta, scale_est, _ = estimate_rotation_scale(template, window, cfg, grid)
        theta_final, shift, _surface, _peak = resolve_rotation(
            template, window, d_theta, scale_est, temperature=cfg.temperature,
            corr_window=cfg.corr_window)

        matrix_true = similarity_matrix_about_center(theta_true, scale_true, tx_true, ty_true, size)
        matrix_est = similarity_matrix_about_center(theta_final, scale_est, shift[:, 0], shift[:, 1], size)
        corner_error = (project_corners(matrix_est) - project_corners(matrix_true)).norm(dim=-1).mean()

        assert corner_error.item() < 8.0, f"case {case} gave corner_error={corner_error.item():.2f}px"


def test_apply_similarity_warp_matches_similarity_matrix_about_center():
    """Las dos representaciones de la MISMA transformación (deformación de
    imagen vs. mapa de puntos) deben ser consistentes: el contenido que
    aparece en `warped` en la posición predicha por
    `similarity_matrix_about_center` debe coincidir con el pico real."""
    theta, scale, tx, ty = torch.tensor([0.4]), torch.tensor([1.3]), torch.tensor([6.0]), torch.tensor([-2.0])
    size = 81
    content = torch.zeros(1, 1, size, size)
    py, px = 20, 25
    content[0, 0, py, px] = 1.0
    warped = apply_similarity_warp(content, theta, scale, tx, ty, out_size=(size, size))
    matrix = similarity_matrix_about_center(theta, scale, tx, ty, size)

    predicted = (matrix[0] @ torch.tensor([float(px), float(py), 1.0]))[:2]
    peak_index = torch.argmax(warped[0, 0])
    actual_y, actual_x = peak_index // size, peak_index % size

    assert abs(predicted[0].item() - actual_x.item()) < 1.5
    assert abs(predicted[1].item() - actual_y.item()) < 1.5


def _tiny_model() -> Task2DPCNModel:
    # Configuración reducida solo para que el smoke test corra rápido en
    # CPU -- las formas/gradientes no dependen de la resolución real
    # (1024px), que se usa en el entrenamiento real sobre GPU.
    return Task2DPCNModel(
        fundus_size=128,
        coarse_channels=(8, 16),
        fine_channels=(8, 16),
        window_size=64.0,
        working_size=32,
        fine_feat_size=16,
        corner_hidden=16,
        log_polar=LogPolarConfig(size=16, radius_bins=12, angle_bins=16, corr_window=5),
    )


def test_model_forward_shapes_and_full_gradient_flow():
    torch.manual_seed(0)
    model = _tiny_model()
    fundus = torch.rand(2, 3, 128, 128)
    enface = torch.rand(2, 1, 64, 32)

    output = model(fundus, enface)

    assert output["pred_matrix"].shape == (2, 3, 3)
    assert output["corners_refined"].shape == (2, 4, 2)
    assert torch.allclose(output["pred_matrix"][:, 2], torch.tensor([0.0, 0.0, 1.0]).expand(2, 3), atol=1e-5)

    output["pred_matrix"].sum().backward()
    missing_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_zero_init_corner_head_does_not_perturb_classical_estimate():
    """La última capa de `corner_head` se inicializa en cero a propósito
    (ver docstring de `Task2DPCNModel`): al arrancar, el delta de esquinas
    debe ser exactamente cero, así que `corners_refined ==
    corners_base` -- el modelo arranca siendo el estimador clásico puro, no
    una perturbación aleatoria de él."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    fundus = torch.rand(2, 3, 128, 128)
    enface = torch.rand(2, 1, 64, 32)
    with torch.no_grad():
        output = model(fundus, enface)
    assert torch.equal(output["corner_delta"], torch.zeros_like(output["corner_delta"]))
    assert torch.equal(output["corners_base"], output["corners_refined"])


def test_valid_mask_shape_is_enforced():
    torch.manual_seed(0)
    model = _tiny_model()
    fundus = torch.rand(1, 3, 128, 128)
    enface = torch.rand(1, 1, 64, 32)
    bad_mask = torch.ones(1, 3, 128, 128, dtype=torch.bool)
    try:
        model(fundus, enface, valid_mask=bad_mask)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for malformed valid_mask")
