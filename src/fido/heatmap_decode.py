"""Decodificación sub-píxel de heatmaps — agnóstico a backbone y a tarea.

Aplica igual a Task 1 (keypoint de instrumento) y a Task 2 si la cabeza de
posición se implementa como heatmap de (tx,ty).

Por qué existe este módulo: literature/02_task1_keypoints.md §3.3 y
literature/03_backbones_recipes.md §4b son explícitos en que el argmax duro
sobre un heatmap de stride alto pierde AUC por cuantización, no por error del
modelo. Con stride 8 y umbrales enteros 0..10 se pierden los primeros 4-5
umbrales SIEMPRE, sin importar qué tan bueno sea el backbone. Con DINOv2
(stride 14 nativo) el problema es peor: heatmap de 37x37 a 518px.

DARK (arXiv:1910.06278) y UDP (arXiv:1911.07524) son las dos técnicas
citadas por ambos informes de literatura de forma independiente.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _spatial_size(size: int | tuple[int, int]) -> tuple[int, int]:
    """Normaliza ``(height, width)``; un entero representa una grilla cuadrada."""
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("spatial size must be an int or (height, width)")
    return int(size[0]), int(size[1])


def image_to_heatmap_xy(
    xy: torch.Tensor,
    image_size: int | tuple[int, int],
    heatmap_size: int | tuple[int, int],
) -> torch.Tensor:
    """Convierte (x,y) con la convención UDP alineada por esquinas.

    UDP usa ``(output_size - 1) / (input_size - 1)``: los centros 0 y N-1
    coinciden exactamente en ambas grillas y no se introduce el sesgo de
    medio stride de la conversión histórica.
    """
    image_h, image_w = _spatial_size(image_size)
    heat_h, heat_w = _spatial_size(heatmap_size)
    if min(image_h, image_w, heat_h, heat_w) <= 1:
        raise ValueError("UDP coordinate conversion requires spatial sizes > 1")
    scale = xy.new_tensor([(heat_w - 1) / (image_w - 1), (heat_h - 1) / (image_h - 1)])
    return xy * scale


def heatmap_to_image_xy(
    xy: torch.Tensor,
    image_size: int | tuple[int, int],
    heatmap_size: int | tuple[int, int],
) -> torch.Tensor:
    """Inversa exacta de :func:`image_to_heatmap_xy`."""
    image_h, image_w = _spatial_size(image_size)
    heat_h, heat_w = _spatial_size(heatmap_size)
    if min(image_h, image_w, heat_h, heat_w) <= 1:
        raise ValueError("UDP coordinate conversion requires spatial sizes > 1")
    scale = xy.new_tensor([(image_w - 1) / (heat_w - 1), (image_h - 1) / (heat_h - 1)])
    return xy * scale


def soft_argmax_2d(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Soft-argmax con temperatura sobre un heatmap (B, C, H, W).

    Devuelve coordenadas (B, C, 2) en el espacio del heatmap, orden (x, y).
    Diferenciable de punta a punta — a diferencia de un argmax duro, permite
    que el gradiente fluya y da subpíxel gratis en la posición del pico.
    """
    batch, channels, height, width = heatmap.shape
    flat = heatmap.reshape(batch, channels, -1)
    weights = F.softmax(flat / temperature, dim=-1).reshape(batch, channels, height, width)

    ys = torch.linspace(0, height - 1, height, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.linspace(0, width - 1, width, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    coord_x = (weights * grid_x).sum(dim=(-2, -1))
    coord_y = (weights * grid_y).sum(dim=(-2, -1))
    return torch.stack([coord_x, coord_y], dim=-1)


def local_soft_argmax_2d(heatmap: torch.Tensor, window: int = 7,
                          temperature: float = 1.0) -> torch.Tensor:
    """Soft-argmax restringido a una ventana local alrededor del argmax duro.

    El soft-argmax GLOBAL es sensible a modos espurios: un segundo instrumento
    en el fundus, un reflejo especular del endoiluminador, la sombra propia
    del instrumento (literature/02_task1_keypoints.md §3.4, recomendación #3).
    Restringir a una ventana evita que esos modos secuestren el promedio
    ponderado, sin perder la diferenciabilidad ni la precisión sub-píxel.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    batch, channels, height, width = heatmap.shape
    flat = heatmap.reshape(batch, channels, -1)
    peak_index = flat.argmax(dim=-1)
    peak_y = (peak_index // width).float()
    peak_x = (peak_index % width).float()

    half = window // 2
    ys = torch.arange(height, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.arange(width, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Máscara de ventana centrada en el pico, una por (batch, channel).
    dist_y = grid_y.view(1, 1, height, width) - peak_y.view(batch, channels, 1, 1)
    dist_x = grid_x.view(1, 1, height, width) - peak_x.view(batch, channels, 1, 1)
    in_window = (dist_y.abs() <= half) & (dist_x.abs() <= half)

    masked = heatmap.masked_fill(~in_window, float("-inf"))
    return soft_argmax_2d(masked, temperature=temperature)


def dark_offset_correction(heatmap: torch.Tensor, peak_xy: torch.Tensor,
                            epsilon: float = 1e-9) -> torch.Tensor:
    """Corrección sub-píxel estilo DARK vía expansión de Taylor de 2º orden
    sobre el log-heatmap en el pico.

    `peak_xy`: (B, C, 2) coordenadas enteras del pico (argmax duro).
    Devuelve el offset (B, C, 2) a sumar a `peak_xy` para la posición corregida.

    Requiere que el heatmap tenga un mínimo de suavizado gaussiano (el GT de
    entrenamiento ya lo tiene si se generó con un kernel gaussiano estándar).
    """
    log_heatmap = torch.log(heatmap.clamp_min(epsilon))
    batch, channels, height, width = log_heatmap.shape

    if height < 3 or width < 3:
        return torch.zeros_like(peak_xy)
    raw_x = peak_xy[..., 0].round().long()
    raw_y = peak_xy[..., 1].round().long()
    interior = (raw_x > 0) & (raw_x < width - 1) & (raw_y > 0) & (raw_y < height - 1)
    x = raw_x.clamp(1, width - 2)
    y = raw_y.clamp(1, height - 2)

    batch_index = torch.arange(batch, device=heatmap.device)[:, None]
    channel_index = torch.arange(channels, device=heatmap.device)[None, :]

    def sample(dy: int, dx: int) -> torch.Tensor:
        return log_heatmap[batch_index, channel_index, y + dy, x + dx]

    center = sample(0, 0)
    right = sample(0, 1)
    left = sample(0, -1)
    up = sample(-1, 0)
    down = sample(1, 0)
    up_right = sample(-1, 1)
    up_left = sample(-1, -1)
    down_right = sample(1, 1)
    down_left = sample(1, -1)

    grad_x = (right - left) / 2.0
    grad_y = (down - up) / 2.0
    grad_xx = right - 2 * center + left
    grad_yy = down - 2 * center + up
    grad_xy = (down_right - down_left - up_right + up_left) / 4.0

    hessian = torch.stack([
        torch.stack([grad_xx, grad_xy], dim=-1),
        torch.stack([grad_xy, grad_yy], dim=-1),
    ], dim=-2)
    gradient = torch.stack([grad_x, grad_y], dim=-1)

    hessian_reg = hessian + epsilon * torch.eye(2, device=heatmap.device, dtype=heatmap.dtype)
    offset = -torch.linalg.solve_ex(hessian_reg, gradient.unsqueeze(-1)).result.squeeze(-1)
    # El offset de Newton diverge si el Hessiano no es negativo-definido (pico
    # plano o en el borde); acotarlo a media celda evita saltos absurdos.
    offset = torch.nan_to_num(offset).clamp(-0.5, 0.5)
    return torch.where(interior.unsqueeze(-1), offset, torch.zeros_like(offset))


def decode_heatmap(
    heatmap: torch.Tensor,
    mode: str,
    temperature: float = 1.0,
    window: int = 7,
) -> torch.Tensor:
    """Despacha decodificadores y devuelve ``(B,C,2)`` en unidades de heatmap.

    La entrada son logits. DARK calcula derivadas sobre probabilidades sigmoid
    positivas; los NaN se convierten en logits muy negativos para que un caso
    corrupto no contamine toda la evaluación.
    """
    if heatmap.ndim != 4:
        raise ValueError("heatmap must have shape (B, C, H, W)")
    logits = torch.nan_to_num(heatmap, nan=-1e9, posinf=1e9, neginf=-1e9)
    if mode == "global":
        return soft_argmax_2d(logits, temperature=temperature)
    if mode == "local":
        return local_soft_argmax_2d(logits, window=window, temperature=temperature)
    if mode == "dark":
        batch, channels, _, width = logits.shape
        peak_index = logits.reshape(batch, channels, -1).argmax(dim=-1)
        peak = torch.stack((peak_index % width, peak_index // width), dim=-1).to(logits.dtype)
        probability = torch.sigmoid(logits)
        return peak + dark_offset_correction(probability, peak)
    raise ValueError(f"unknown decoder mode: {mode!r}")


def gaussian_heatmap_target(center_xy: torch.Tensor, height: int, width: int,
                             sigma: float, device=None, dtype=None) -> torch.Tensor:
    """Genera el heatmap gaussiano de entrenamiento para una posición dada.

    `center_xy`: (..., 2) en coordenadas de píxel del heatmap (puede ser
    fraccionario — no se cuantiza, evitando el sesgo que documenta
    literature/02_task1_keypoints.md §3.2 sobre codificar el GT con enteros).
    """
    device = device or center_xy.device
    dtype = dtype or center_xy.dtype
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    center_x = center_xy[..., 0].unsqueeze(-1).unsqueeze(-1)
    center_y = center_xy[..., 1].unsqueeze(-1).unsqueeze(-1)
    squared_dist = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    return torch.exp(-squared_dist / (2 * sigma**2))
