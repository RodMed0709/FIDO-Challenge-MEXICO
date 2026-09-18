"""DPCN/DPCN++-style registration for Task 2: log-polar phase correlation en
dominio de features aprendidas, con búsqueda de ventana candidata previa
(coarse-to-fine) y una cabeza de 4 esquinas (DeTone) con ajuste cerrado.

Por qué esta arquitectura y no una regresión directa (`task2_baseline.py`,
`task2_common.py`): T2-R1..T2-R6 en ATTACK_LADDER.md midieron, no
argumentaron, que la correlación cruzada espacial simple no despega (AUC
0.00). Esta corrida (T2-92) mide explícitamente la familia DPCN/DPCN++
(Zhao et al. 2019/2021, correlación de fase diferenciable en log-polar), que
ningún peldaño anterior probó. Se implementa DENSA y POTENTE a propósito
(encargo de Rodrigo): si esto tampoco despega, el negativo es concluyente
sobre la familia de método, no sobre falta de capacidad.

## El reto central: solapamiento minúsculo

La huella del OCT es ~2.4% del área del fundus (15.6% de lado). DPCN++
(sección 6.4 de su paper) reporta degradación con solapamiento pequeño
porque corre la correlación en log-polar sobre EL PAR COMPLETO de imágenes
-- con una plantilla que ocupa un recorte diminuto del mapa de búsqueda, el
espectro de magnitud de Fourier del fundus completo está dominado por
contenido que NO es la huella, y la correlación de fase pierde SNR.

Esta implementación ataca el problema explícitamente con un esquema de dos
etapas:

1. **Localización gruesa** (`_coarse_localize`): correlación cruzada densa
   estilo SiamFC (mismo truco que `task2_baseline.py`: grouped conv2d,
   soft-argmax) entre encoders DENSOS por modalidad, entrenada de punta a
   punta, que da un centro candidato en el fundus completo (1024px). Esto
   reduce el problema de "encontrar una aguja en un pajar" a "recortar una
   ventana alrededor de una estimación aproximada".
2. **Registración fina** (`_fine_register`): SOLO dentro de una ventana
   recortada (differentiable crop vía `affine_grid`/`grid_sample`, tamaño
   `window_size` en píxeles del fundus completo -- generoso respecto a la
   huella esperada, cubriendo el rango de escala aumentado). Dentro de esa
   ventana, mucho más comparable en escala a la plantilla OCT, se corre el
   núcleo DPCN real: magnitud de Fourier -> log-polar diferenciable ->
   correlación de fase diferenciable (rotación+escala) -> alineación ->
   correlación de fase espacial (traslación).

Este diseño ataca el argumento central de la sección 6.4 de DPCN++ por
construcción: la correlación de fase nunca ve el fundus completo, solo la
ventana candidata.

## Ambigüedad de 180 grados (Fourier-Mellin clásico, no inventada)

El espectro de magnitud de una imagen real es simétrico por punto
(|F(-u,-v)| = |F(u,v)|), así que el log-polar de la magnitud tiene período
angular pi, no 2*pi -- la correlación de fase en ese dominio solo recupera
theta módulo pi. Se resuelve con la técnica estándar de Fourier-Mellin:
evaluar las dos rotaciones candidatas (theta, theta+pi) y quedarse con la
que da un pico de correlación de traslación más alto (`_resolve_rotation`).
Verificado con casos sintéticos en `tests/test_task2_dpcn.py`.

## Marco canónico: por qué NO se usa `fit_reflected_similarity`

El encargo original pedía reutilizar `fit_reflected_similarity` de
`analysis/run_task2_instrument_gate.py` (que impone A=[[a,b],[b,-a]],
det<0) para evitar el bug de ambigüedad de signo de
`fit_closed_form_similarity` documentado ahí. Esa función opera en el marco
NATIVO del JSON (`Ground Truth/Task 2`), donde la relación real SÍ es una
similitud reflejada (confirmado en el 100% de los 1214 casos, ver
`fido-task2-geometry.md`).

Pero todo el pipeline de datos de Task 2 (`fido.data.task2.Task2Dataset`,
usado también por `task2_baseline.py` y `task2_common.py`) ya canonicaliza
el en-face (`canonicalize_task2_enface`, convención
`transpose__flip_u__flip_v`) y compone el GT nativo con la matriz fija `C`
de esa convención (`canonicalize_task2_matrix`). `C` tiene det=-1, así que
`M_native_reflejada @ C` es una similitud PROPIA (det>0) en el marco
canónico -- exactamente lo que ya hace `Task2Dataset._load_case` al llamar
`decompose_similarity(gt_matrix, reflect=False)`, y lo que
`train_task2_baseline.py` asume en `compose_similarity(..., reflect=False)`.

Este módulo opera enteramente en ese mismo marco canónico (misma convención
que el resto del código de Task 2). Ahí la matriz objetivo NO está
reflejada -- imponer la estructura de `fit_reflected_similarity` (A=[[a,b],
[b,-a]]) a datos que son en realidad A=[[a,-b],[b,a]] no evitaría el bug de
ambigüedad de signo, LO CAUSARÍA: el ajuste ya no podría ser exacto ni con
datos sin ruido, porque la estructura impuesta sería la incorrecta para
este marco. `fit_similarity_from_corners` (más abajo) implementa el ajuste
lineal análogo pero con el signo correcto para el marco canónico
(A=[[a,-b],[b,a]], det>0, misma familia que `compose_similarity(...,
reflect=False)` y que la propia `geometry.fit_closed_form_similarity`,
salvo que aquí se deriva como una matriz PSEUDOINVERSA FIJA -- ref_uv son
siempre las 4 esquinas del cuadrado unitario, constantes -- para que la
cabeza de esquinas sea una única matmul diferenciable por lote, no un SVD
por muestra). Verificado exacto (~1e-5 px) en
`test_fit_similarity_from_corners_recovers_known_transform`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fido.geometry import UNIT_SQUARE_CORNERS, compose_similarity, decompose_similarity, project_corners
from fido.heatmap_decode import local_soft_argmax_2d, soft_argmax_2d

# Mismo valor que SCALE_REF en task2_baseline.py: escala media medida sobre
# los 1214 casos reales de entrenamiento (160 +- 13.7 px, ver T2-R1). Solo
# fija el tamaño de la plantilla en la etapa GRUESA de localización (para
# que kernel y mapa de búsqueda queden en la misma escala tras el stride del
# encoder, mismo razonamiento que en `task2_baseline.py`); la etapa FINA no
# depende de este valor.
COARSE_TEMPLATE_REF_PX = 160.0


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = _num_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseModalityEncoder(nn.Module):
    """Encoder CNN denso por modalidad -- SIN pesos compartidos entre fundus
    y OCT (son modalidades distintas, encargo explícito). Cada instancia de
    este modelo se crea por separado para cada rama y cada etapa (gruesa vs
    fina), dando capacidad de sobra a propósito: si esto no basta para que
    DPCN despegue, la conclusión es sobre el método, no sobre el tamaño de
    la red.
    """

    def __init__(self, in_channels: int, channels: tuple[int, ...] = (48, 96, 192, 384),
                 out_spatial: tuple[int, int] | None = None) -> None:
        super().__init__()
        stages = []
        current = in_channels
        for out_channels in channels:
            stages.append(_ConvBlock(current, out_channels, stride=2))
            current = out_channels
        self.stages = nn.Sequential(*stages)
        self.out_channels = current
        self.out_spatial = out_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stages(x)
        if self.out_spatial is not None:
            feat = F.adaptive_avg_pool2d(feat, self.out_spatial)
        return feat


# --------------------------------------------------------------------------
# Núcleo Fourier-Mellin diferenciable: funciones puras, sin estado de red.
# Testeadas de forma aislada con casos sintéticos de respuesta conocida.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LogPolarConfig:
    """Configuración fija del muestreo log-polar. `size` es el lado (S) de
    las imágenes cuadradas de entrada (plantilla y ventana, YA en la misma
    resolución de trabajo). Ángulo cubre SOLO [0, pi) -- el espectro de
    magnitud de una señal real es simétrico por punto (periodo pi, no
    2*pi), cubrir [0, 2*pi) duplicaría cada muestra sin dar información
    nueva. `radius_bins`/`angle_bins` = (96, 192) y `temperature`/`window`
    = (0.03, 9) son los valores que sobrevivieron una barrida sintética
    (ver ATTACK_LADDER.md T2-92): con radius_bins mucho menor que
    angle_bins la correlación log-polar se vuelve inestable (falsos picos),
    documentado explícitamente para no repetir la búsqueda a ciegas.
    """

    size: int
    radius_bins: int = 96
    angle_bins: int = 192
    r_min: float = 2.0
    r_max_frac: float = 0.45  # fracción de size/2
    temperature: float = 0.03
    corr_window: int = 9

    @property
    def r_max(self) -> float:
        return self.r_max_frac * self.size / 2.0

    @property
    def log_r_min(self) -> float:
        return math.log(self.r_min)

    @property
    def log_r_max(self) -> float:
        return math.log(self.r_max)


def build_log_polar_grid(cfg: LogPolarConfig, device=None, dtype=torch.float32) -> torch.Tensor:
    """Grid fijo (radius_bins, angle_bins, 2) para `F.grid_sample`
    (align_corners=True). No depende de los datos -- se calcula una vez y
    se registra como buffer del módulo que lo usa."""
    s = cfg.size
    center = (s - 1) / 2.0
    log_r = torch.linspace(cfg.log_r_min, cfg.log_r_max, cfg.radius_bins, device=device, dtype=dtype)
    radii = torch.exp(log_r)
    angles = torch.arange(cfg.angle_bins, device=device, dtype=dtype) * (math.pi / cfg.angle_bins)
    rr, aa = torch.meshgrid(radii, angles, indexing="ij")
    xs = center + rr * torch.cos(aa)
    ys = center + rr * torch.sin(aa)
    xs_n = 2.0 * xs / (s - 1) - 1.0
    ys_n = 2.0 * ys / (s - 1) - 1.0
    return torch.stack([xs_n, ys_n], dim=-1)


def log_polar_warp(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """x: (B,C,S,S). grid: (radius_bins, angle_bins, 2). ->
    (B,C,radius_bins,angle_bins). Diferenciable en `x` (grid es fijo)."""
    batch = x.shape[0]
    sampling_grid = grid.unsqueeze(0).expand(batch, -1, -1, -1).to(dtype=x.dtype)
    return F.grid_sample(x, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def fourier_log_magnitude(x: torch.Tensor) -> torch.Tensor:
    """log(1+|FFT2|), con la frecuencia cero centrada (`fftshift`) -- la
    representación clásica de Fourier-Mellin, invariante a traslación de
    `x`. FFT se calcula siempre en float32 (CPU/CUDA no soportan bien FFT en
    float16/bf16); el resultado vuelve al dtype de entrada."""
    spectrum = torch.fft.fft2(x.float())
    magnitude = torch.fft.fftshift(spectrum.abs(), dim=(-2, -1))
    return torch.log1p(magnitude).to(dtype=x.dtype)


def hann_window_2d(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Ventana de Hann 2D separable -- reduce el "leakage" espectral del
    borde finito de la imagen antes de la FFT (práctica estándar de
    Fourier-Mellin). Se aplica en el dominio ESPACIAL, después de cualquier
    rotación/escala (nunca antes: una ventana separable no es invariante a
    rotación, aplicarla antes de rotar sesga el centroide -- verificado
    empíricamente, ver ATTACK_LADDER.md T2-92)."""
    wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    return torch.outer(wy, wx)


def phase_correlation(a: torch.Tensor, b: torch.Tensor, *, temperature: float = 1.0,
                       window: int = 9, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Correlación de fase diferenciable vía espectro de potencia cruzada
    normalizado. a, b: (B,C,H,W) reales, mismo tamaño.

    Convención verificada con un caso sintético exacto (ver
    `test_phase_correlation_recovers_known_shift`): si el contenido de `b`
    es el de `a` desplazado por `shift` (b = roll(a, shift)), esta función
    devuelve `shift` (no `-shift`).

    Multicanal: cada canal aporta evidencia independiente para el MISMO
    desplazamiento (misma transformación geométrica subyacente), así que
    las superficies de correlación por canal se promedian tras la IFFT,
    antes de extraer el pico -- no se promedian los espectros (eso
    mezclaría fase entre canales de forma incorrecta).

    Devuelve (shift_xy: (B,2), surface: (B,1,H,W) ya con `fftshift`, útil
    como diagnóstico o como entrada de la cabeza de refinamiento).
    """
    if a.shape != b.shape:
        raise ValueError("phase_correlation requires equally-shaped inputs")
    fa = torch.fft.fft2(a.float())
    fb = torch.fft.fft2(b.float())
    cross = fa * torch.conj(fb)
    cross = cross / (cross.abs() + eps)
    corr = torch.fft.ifft2(cross).real.mean(dim=1, keepdim=True)
    corr = torch.fft.fftshift(corr, dim=(-2, -1)).to(dtype=a.dtype)
    coords = local_soft_argmax_2d(corr, window=window, temperature=temperature)  # (B,1,2) xy
    height, width = corr.shape[-2:]
    center = corr.new_tensor([width // 2, height // 2])
    shift = center - coords[:, 0, :]
    return shift, corr


def similarity_matrix_about_center(theta: torch.Tensor, scale: torch.Tensor, tx: torch.Tensor,
                                    ty: torch.Tensor, size: int) -> torch.Tensor:
    """Matriz 3x3 (B,3,3) de la similitud PROPIA (sin reflexión, misma
    familia que `compose_similarity(..., reflect=False)`) que rota por
    `theta` y escala por `scale` alrededor del CENTRO de un lienzo
    cuadrado de lado `size`, y luego traslada por (tx,ty). Mapea puntos en
    píxeles del lienzo de entrada a píxeles del lienzo de salida (misma
    resolución `size`).
    """
    center = (size - 1) / 2.0
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    a = scale * cos_t
    b = scale * sin_t
    tx_full = center * (1.0 - a + b) + tx
    ty_full = center * (1.0 - a - b) + ty
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    row0 = torch.stack([a, -b, tx_full], dim=-1)
    row1 = torch.stack([b, a, ty_full], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def apply_similarity_warp(x: torch.Tensor, theta: torch.Tensor, scale: torch.Tensor,
                           tx: torch.Tensor | None = None, ty: torch.Tensor | None = None,
                           out_size: tuple[int, int] | None = None) -> torch.Tensor:
    """Deforma el CONTENIDO de `x` (B,C,H,W) rotándolo `theta` (rad) y
    escalándolo `scale` alrededor de su centro, más una traslación opcional
    (tx,ty) en píxeles de salida. Convención verificada con
    `test_apply_similarity_warp_matches_similarity_matrix_about_center`:
    el contenido de la salida en el punto `similarity_matrix_about_center(
    theta, scale, tx, ty, size) @ p` es el de `x` en `p` -- es decir, esta
    función y `similarity_matrix_about_center` describen la MISMA
    transformación, una como deformación de imagen (para alinear
    tensores) y otra como mapa de puntos (para componer homografías).
    """
    batch, channels, height, width = x.shape
    out_h, out_w = out_size or (height, width)
    if tx is None:
        tx = torch.zeros_like(theta)
    if ty is None:
        ty = torch.zeros_like(theta)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    inv_scale = 1.0 / scale.clamp_min(1e-6)
    a = inv_scale * cos_t
    s = inv_scale * sin_t
    size = x.shape[-1]
    tx_n = tx * 2.0 / max(size - 1, 1)
    ty_n = ty * 2.0 / max(size - 1, 1)
    tx_in = -(a * tx_n + s * ty_n)
    ty_in = -(-s * tx_n + a * ty_n)
    theta_mat = torch.stack([
        torch.stack([a, s, tx_in], dim=-1),
        torch.stack([-s, a, ty_in], dim=-1),
    ], dim=-2)
    grid = F.affine_grid(theta_mat, (batch, channels, out_h, out_w), align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def estimate_rotation_scale(template_feat: torch.Tensor, window_feat: torch.Tensor,
                             cfg: LogPolarConfig, grid: torch.Tensor):
    """Núcleo Fourier-Mellin: recupera (d_theta, scale) tal que el contenido
    de `window_feat` es aproximadamente el de `template_feat` rotado por
    `d_theta` y escalado por `scale` (ver `similarity_matrix_about_center`).
    `d_theta` está módulo pi (ambigüedad de 180 grados de Fourier-Mellin
    clásico, ver docstring del módulo) -- resolver con `resolve_rotation`.
    template_feat, window_feat: (B,C,S,S), S=cfg.size.
    """
    template_mag = fourier_log_magnitude(template_feat)
    window_mag = fourier_log_magnitude(window_feat)
    template_lp = log_polar_warp(template_mag, grid)
    window_lp = log_polar_warp(window_mag, grid)
    shift, surface = phase_correlation(template_lp, window_lp, temperature=cfg.temperature,
                                       window=cfg.corr_window)
    d_angle_idx, d_radius_idx = shift[:, 0], shift[:, 1]
    d_theta = d_angle_idx * (math.pi / cfg.angle_bins)
    d_logr = d_radius_idx * (cfg.log_r_max - cfg.log_r_min) / (cfg.radius_bins - 1)
    scale = torch.exp(-d_logr)
    return d_theta, scale, surface


def resolve_rotation(template_feat: torch.Tensor, window_feat: torch.Tensor,
                      d_theta: torch.Tensor, scale: torch.Tensor, *,
                      temperature: float = 0.03, corr_window: int = 9):
    """Desambigua la rotación módulo pi evaluando las dos candidatas
    (d_theta, d_theta+pi): alinea la plantilla con cada una y compara el
    pico de la correlación de fase de TRASLACIÓN -- la candidata correcta
    da un pico más nítido (técnica estándar de Fourier-Mellin, no
    inventada). Devuelve (theta_final, translation_shift, surface,
    peak_value) para la candidata ganadora.
    """
    size = template_feat.shape[-1]
    best = None
    for candidate in (d_theta, d_theta + math.pi):
        aligned = apply_similarity_warp(template_feat, candidate, scale, out_size=(size, size))
        hann = hann_window_2d(size, size, device=aligned.device, dtype=aligned.dtype)
        shift, surface = phase_correlation(aligned * hann, window_feat * hann,
                                           temperature=temperature, window=corr_window)
        peak_value = surface.amax(dim=(-2, -1))  # (B,1)
        if best is None:
            best = (candidate, shift, surface, peak_value)
        else:
            take_new = (peak_value > best[3]).view(-1)
            new_theta = torch.where(take_new, candidate, best[0])
            new_shift = torch.where(take_new.unsqueeze(-1), shift, best[1])
            new_peak = torch.where(take_new.view(-1, 1), peak_value, best[3])
            best = (new_theta, new_shift, surface, new_peak)
    theta_final, shift, surface, peak_value = best
    return theta_final, shift, surface, peak_value


# --------------------------------------------------------------------------
# Cabeza de esquinas (DeTone) + ajuste cerrado a similitud propia.
# --------------------------------------------------------------------------

def _corner_fit_pinv() -> torch.Tensor:
    """Pseudoinversa FIJA (4,8) del sistema lineal que ajusta (a,b,tx,ty)
    -- con A=[[a,-b],[b,a]], misma familia que `compose_similarity(...,
    reflect=False)` -- a partir de 4 correspondencias esquina-a-esquina
    con las esquinas de referencia del cuadrado unitario
    (`UNIT_SQUARE_CORNERS`), que son SIEMPRE las mismas 4 constantes. Por
    eso la pseudoinversa se puede precalcular una sola vez: el ajuste por
    lote se vuelve una única matmul diferenciable, sin SVD ni resolver un
    sistema por muestra.
    """
    ref = UNIT_SQUARE_CORNERS[:, :2]
    design = np.zeros((8, 4), dtype=np.float64)
    for i, (u, v) in enumerate(ref):
        design[2 * i] = [u, -v, 1.0, 0.0]
        design[2 * i + 1] = [v, u, 0.0, 1.0]
    return torch.from_numpy(np.linalg.pinv(design)).to(torch.float32)


def fit_similarity_from_corners(corners: torch.Tensor, pinv: torch.Tensor) -> torch.Tensor:
    """corners: (B,4,2) en el mismo orden que `UNIT_SQUARE_CORNERS`
    (esquinas predichas del cuadrado unitario, en píxeles del fundus).
    pinv: buffer (4,8) de `_corner_fit_pinv()`. Devuelve la matriz 3x3
    (B,3,3) de la similitud propia que mejor ajusta esas 4
    correspondencias en mínimos cuadrados -- exacta (sin residuo) si las 4
    esquinas provienen realmente de una similitud propia, como en el marco
    canónico de Task 2.
    """
    batch = corners.shape[0]
    target = corners.reshape(batch, 8)
    params = target @ pinv.to(dtype=corners.dtype, device=corners.device).T  # (B,4): a,b,tx,ty
    a, b, tx, ty = params.unbind(dim=-1)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    row0 = torch.stack([a, -b, tx], dim=-1)
    row1 = torch.stack([b, a, ty], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# --------------------------------------------------------------------------
# Recorte diferenciable de ventana candidata (localización gruesa -> fina).
# --------------------------------------------------------------------------

def batched_crop_resize(image: torch.Tensor, center_xy: torch.Tensor, crop_size: float,
                         out_size: tuple[int, int]) -> torch.Tensor:
    """Recorta un cuadrado de lado `crop_size` (píxeles de `image`) centrado
    en `center_xy` (B,2) por muestra, y lo remuestrea a `out_size`.
    Diferenciable en `image` y en `center_xy` (permite entrenar la
    localización gruesa con el gradiente de la etapa fina). Sin rotación --
    puro recorte+resize, igual convención que
    `pixel_crop_resize_homography` en `task2_transforms.py` pero batched."""
    batch, _, height, width = image.shape
    out_h, out_w = out_size
    cx, cy = center_xy[:, 0], center_xy[:, 1]
    scale_x = (crop_size - 1) / max(width - 1, 1)
    scale_y = (crop_size - 1) / max(height - 1, 1)
    offset_x = 2.0 * cx / max(width - 1, 1) - 1.0
    offset_y = 2.0 * cy / max(height - 1, 1) - 1.0
    zero = torch.zeros_like(cx)
    theta_mat = torch.stack([
        torch.stack([torch.full_like(cx, scale_x), zero, offset_x], dim=-1),
        torch.stack([zero, torch.full_like(cy, scale_y), offset_y], dim=-1),
    ], dim=-2)
    grid = F.affine_grid(theta_mat, (batch, image.shape[1], out_h, out_w), align_corners=True)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def crop_homography(center_xy: torch.Tensor, crop_size: float, out_size: int) -> torch.Tensor:
    """Homografía (B,3,3) que mapea puntos en píxeles de la imagen COMPLETA
    de entrada al grid `out_size x out_size` producido por
    `batched_crop_resize` con los mismos argumentos -- versión batched de
    `pixel_crop_resize_homography` (misma fórmula, ver
    `task2_transforms.py`), para componer la matriz de similitud
    recuperada con el recorte y devolverla en coordenadas del fundus
    completo.
    """
    cx, cy = center_xy[:, 0], center_xy[:, 1]
    x = cx - crop_size / 2.0
    y = cy - crop_size / 2.0
    scale = out_size / crop_size
    zero = torch.zeros_like(cx)
    one = torch.ones_like(cx)
    row0 = torch.stack([torch.full_like(cx, scale), zero, -x * scale], dim=-1)
    row1 = torch.stack([zero, torch.full_like(cy, scale), -y * scale], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# --------------------------------------------------------------------------
# Modelo completo.
# --------------------------------------------------------------------------

class Task2DPCNModel(nn.Module):
    """Pipeline completo: localización gruesa (SiamFC denso) -> recorte de
    ventana candidata -> registración fina (DPCN: log-polar + correlación
    de fase diferenciable) -> cabeza de 4 esquinas + ajuste cerrado.

    Devuelve un dict con `pred_matrix` (B,3,3, marco canónico, familia
    `reflect=False`) y diagnósticos intermedios (`coarse_center`,
    `theta`, `scale`, `corners_base`, `corners_refined`, superficies de
    correlación) -- útiles para pérdidas auxiliares y para el control
    OCT-shuffle (comparar `pred_matrix` con el en-face de otro caso
    barajado en el batch).
    """

    def __init__(self, *, fundus_size: int = 1024,
                 coarse_channels: tuple[int, ...] = (48, 96, 192, 384),
                 fine_channels: tuple[int, ...] = (32, 64, 128),
                 window_size: float = 512.0,
                 working_size: int = 256,
                 fine_feat_size: int = 128,
                 corner_hidden: int = 256,
                 log_polar: LogPolarConfig | None = None) -> None:
        super().__init__()
        self.fundus_size = fundus_size
        self.window_size = window_size
        self.working_size = working_size
        self.fine_feat_size = fine_feat_size

        # --- etapa gruesa: localización aproximada sobre el fundus completo ---
        self.coarse_fundus_encoder = DenseModalityEncoder(3, coarse_channels)
        self.coarse_oct_encoder = DenseModalityEncoder(1, coarse_channels)

        # --- etapa fina: DPCN dentro de la ventana candidata ---
        self.fine_fundus_encoder = DenseModalityEncoder(
            3, fine_channels, out_spatial=(fine_feat_size, fine_feat_size))
        self.fine_oct_encoder = DenseModalityEncoder(
            1, fine_channels, out_spatial=(fine_feat_size, fine_feat_size))

        cfg = log_polar or LogPolarConfig(size=fine_feat_size)
        if cfg.size != fine_feat_size:
            raise ValueError("log_polar.size must match fine_feat_size")
        self.log_polar_cfg = cfg
        self.register_buffer("log_polar_grid", build_log_polar_grid(cfg), persistent=False)
        self.register_buffer("corner_fit_pinv", _corner_fit_pinv(), persistent=False)

        # Cabeza de refinamiento de esquinas: toma evidencia agregada
        # (features gruesas de ambas ramas + pico de las dos correlaciones
        # de fase) y corrige las 4 esquinas base con un delta APRENDIDO.
        # Última capa inicializada en CERO: al arrancar, el modelo es
        # exactamente el estimador clásico Fourier-Mellin + ajuste cerrado
        # (verificable con datos sintéticos, ver
        # `test_pipeline_zero_init_head_matches_classical_estimate`); el
        # entrenamiento aprende a corregir sesgos sistemáticos del núcleo
        # clásico sin arrancar perturbándolo al azar.
        feat_dim = fine_channels[-1]
        head_in = 2 * feat_dim + 4  # pooled window + pooled template + (cos,sin,log_scale,peak)
        self.corner_head = nn.Sequential(
            nn.Linear(head_in, corner_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(corner_hidden, corner_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(corner_hidden, 8),
        )
        nn.init.zeros_(self.corner_head[-1].weight)
        nn.init.zeros_(self.corner_head[-1].bias)

    # -- etapa gruesa ------------------------------------------------------
    def _coarse_localize(self, fundus: torch.Tensor, enface: torch.Tensor,
                          valid_mask: torch.Tensor) -> torch.Tensor:
        fundus_feat = self.coarse_fundus_encoder(fundus * valid_mask.to(fundus.dtype))
        enface_ref = F.interpolate(enface, size=(int(COARSE_TEMPLATE_REF_PX), int(COARSE_TEMPLATE_REF_PX)),
                                   mode="bilinear", align_corners=False)
        enface_feat = self.coarse_oct_encoder(enface_ref)

        batch, channels, feat_h, feat_w = fundus_feat.shape
        _, _, kh, kw = enface_feat.shape
        pad_h, pad_w = kh // 2, kw // 2
        search = fundus_feat.reshape(1, batch * channels, feat_h, feat_w)
        corr = F.conv2d(search, enface_feat, groups=batch, padding=(pad_h, pad_w))
        out_h, out_w = feat_h + 2 * pad_h - kh + 1, feat_w + 2 * pad_w - kw + 1
        heatmap = corr.reshape(batch, 1, out_h, out_w) / (channels ** 0.5)
        heatmap = (heatmap - heatmap.mean(dim=(2, 3), keepdim=True)) / (
            heatmap.std(dim=(2, 3), keepdim=True) + 1e-6)

        coords = soft_argmax_2d(heatmap, temperature=1.0)[:, 0, :]  # (B,2) xy en celdas
        stride = self.fundus_size / feat_h
        # Mismo -0.5 que en task2_baseline.py: kernel de lado par, la celda
        # de salida `o` corresponde al centro `o - 0.5` bajo padding=(k//2).
        center_x = (coords[:, 0] - 0.5) * stride
        center_y = (coords[:, 1] - 0.5) * stride
        return torch.stack([center_x, center_y], dim=-1)

    # -- etapa fina ----------------------------------------------------------
    def _fine_register(self, fundus: torch.Tensor, enface: torch.Tensor, coarse_center: torch.Tensor):
        window = batched_crop_resize(fundus, coarse_center, self.window_size,
                                     (self.working_size, self.working_size))
        template_input = F.interpolate(enface, size=(self.working_size, self.working_size),
                                       mode="bilinear", align_corners=False)

        window_feat = self.fine_fundus_encoder(window)
        template_feat = self.fine_oct_encoder(template_input)

        d_theta, scale, _rot_surface = estimate_rotation_scale(
            template_feat, window_feat, self.log_polar_cfg, self.log_polar_grid)
        theta, translation, trans_surface, peak_value = resolve_rotation(
            template_feat, window_feat, d_theta, scale,
            temperature=self.log_polar_cfg.temperature, corr_window=self.log_polar_cfg.corr_window)

        # Matriz plantilla_recortada(feat) -> ventana(feat), luego compuesta
        # con las homografías fijas de recorte para volver a coordenadas
        # completas del fundus.
        matrix_feat = similarity_matrix_about_center(theta, scale, translation[:, 0], translation[:, 1],
                                                      self.fine_feat_size)
        window_to_feat = crop_homography(coarse_center, self.window_size, self.fine_feat_size)
        # template_input (working_size) -> template_feat (fine_feat_size):
        # resize isotrópico fijo, sin recorte -- misma fórmula que
        # crop_homography con center = centro del lienzo y crop=working_size.
        template_center = template_input.new_tensor(
            [(self.working_size - 1) / 2.0, (self.working_size - 1) / 2.0]
        ).unsqueeze(0).expand(fundus.shape[0], -1)
        template_to_feat = crop_homography(template_center, self.working_size, self.fine_feat_size)
        # uv del enface canónico (todo el en-face es el cuadrado unitario
        # completo) -> píxeles de `template_input`: resize isotrópico fijo.
        uv_to_template_input = template_input.new_tensor(
            [[self.working_size - 1.0, 0.0, 0.0], [0.0, self.working_size - 1.0, 0.0], [0.0, 0.0, 1.0]]
        ).unsqueeze(0).expand(fundus.shape[0], -1, -1)

        uv_to_feat = template_to_feat @ uv_to_template_input
        matrix_uv_to_window_feat = matrix_feat @ uv_to_feat
        matrix_full = torch.linalg.inv(window_to_feat) @ matrix_uv_to_window_feat

        return {
            "matrix_full": matrix_full,
            "theta": theta,
            "scale": scale,
            "window_feat": window_feat,
            "template_feat": template_feat,
            "peak_value": peak_value.reshape(-1),
            "window": window,
        }

    def forward(self, fundus: torch.Tensor, enface: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = torch.ones_like(fundus[:, :1], dtype=torch.bool)
        if valid_mask.shape != fundus[:, :1].shape:
            raise ValueError("valid_mask must have shape (B,1,H,W)")

        coarse_center = self._coarse_localize(fundus, enface, valid_mask)
        fine = self._fine_register(fundus, enface, coarse_center)

        corners_base = project_corners(fine["matrix_full"])  # (B,4,2)

        window_vec = fine["window_feat"].mean(dim=(2, 3))
        template_vec = fine["template_feat"].mean(dim=(2, 3))
        log_scale = torch.log(fine["scale"].clamp_min(1e-6))
        head_input = torch.cat([
            window_vec, template_vec,
            torch.stack([torch.cos(fine["theta"]), torch.sin(fine["theta"]), log_scale,
                        fine["peak_value"]], dim=-1),
        ], dim=-1)
        corner_delta = self.corner_head(head_input).reshape(-1, 4, 2)
        corners_refined = corners_base + corner_delta

        pred_matrix = fit_similarity_from_corners(corners_refined, self.corner_fit_pinv)
        params = decompose_similarity(pred_matrix, reflect=False)

        return {
            "pred_matrix": pred_matrix,
            "corners_base": corners_base,
            "corners_refined": corners_refined,
            "corner_delta": corner_delta,
            "coarse_center": coarse_center,
            "theta": fine["theta"],
            "scale": fine["scale"],
            "peak_value": fine["peak_value"],
            "tx": params["tx"], "ty": params["ty"],
            "cos_theta": params["cos_theta"], "sin_theta": params["sin_theta"],
            "scale_final": params["scale"],
        }
