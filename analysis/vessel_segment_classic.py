#!/usr/bin/env python3
"""Segmentación clásica de vasos sobre el fundus (canal verde), con y sin
corrección de viñeteado -- pieza añadida al experimento 102 por pedido de
Rodrigo (2026-08-19): si la búsqueda densa con máscaras GT funciona, este
módulo mide cuánto se degrada al pasar de la máscara GT del fundus a una
segmentación DERIVADA DE LA IMAGEN, que es lo único disponible en despliegue
real (Codabench no da la máscara GT de vasos como entrada).

Motivación del viñeteado (`experiments/97-t2-crosshair-recheck/INFORME.md`):
el fundus tiene una caída de brillo radial fuerte (brillo medio 92->0 entre
r=0 y r=600px desde el centro óptico), medida como el causante de un falso
positivo espectacular (Cohen's d=1.98) en un test no relacionado. Cualquier
segmentación por intensidad hereda ese sesgo: los vasos cerca del borde
quedan casi negros y desaparecen. Se implementan dos correcciones clásicas
para probar si mitigan el efecto:

- `correct_divide`: flat-field -- divide por una versión muy suavizada
  (Gaussian de sigma grande) del mismo canal, aproximando el fondo lento
  que causa el viñeteado sin tocar la estructura fina de los vasos.
- `correct_clahe`: CLAHE (contrast-limited adaptive histogram equalization)
  -- normaliza contraste LOCAL en ventanas pequeñas, con lo que el
  gradiente lento del viñeteado deja de importar por construcción.

Método de vesselness: filtro de Frangi (`skimage.filters.frangi`), diseñado
para estructuras tubulares -- estándar en segmentación de vasos retinianos
clásica (pre-deep-learning). Alternativa top-hat morfológico incluida como
comparación más barata/simple.

100% CPU (numpy + scipy + scikit-image + opencv). No requiere GPU.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import frangi


def green_channel(fundus_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (H,W) float32 en [0,1], canal verde. El canal
    verde es estándar en literatura de segmentación de vasos retinianos:
    máximo contraste vaso/fondo de los tres canales RGB."""
    return fundus_rgb[..., 1].astype(np.float32) / 255.0


def correct_divide(img: np.ndarray, sigma: float = 180.0) -> np.ndarray:
    """Flat-field: divide por una estimación suave del fondo (Gaussian de
    sigma grande, del orden de la escala de caída del viñeteado medida en
    T2-crosshair-recheck: ~600px de radio de caída completa -> sigma
    ~180px captura esa escala sin borrar estructura de vaso, que es
    <=15px de ancho típico)."""
    background = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = img / np.maximum(background, 1e-3)
    corrected = corrected - corrected.min()
    corrected = corrected / max(corrected.max(), 1e-6)
    return corrected.astype(np.float32)


def correct_clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 16) -> np.ndarray:
    """CLAHE sobre el canal verde en [0,255] uint8, vía OpenCV."""
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    out = clahe.apply(img_u8)
    return out.astype(np.float32) / 255.0


def frangi_vessel_mask(img: np.ndarray, scales=range(1, 6), threshold_percentile: float = 97.0,
                        visible_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vesselness de Frangi + umbral por percentil. Vasos son estructuras
    OSCURAS sobre fondo claro en el fundus real (no al revés) -- se invierte
    la imagen antes de Frangi, que por convención de skimage realza
    estructuras tubulares CLARAS sobre fondo oscuro (`black_ridges=True` es
    el default y ya asume eso; se deja explícito para que quede documentado
    y no dependa de un default que puede cambiar de versión).

    `visible_mask`: si se pasa (p.ej. círculo de campo visible del fundus),
    el vesselness se pone a 0 fuera de esa región antes de umbralizar --
    evita que el borde oscuro/artefactos fuera de campo cuenten como vaso.
    Devuelve (vesselness_float, mask_binaria_uint8).
    """
    vesselness = frangi(img, sigmas=scales, black_ridges=True)
    if visible_mask is not None:
        vesselness = vesselness * visible_mask.astype(np.float32)
        valid = vesselness[visible_mask > 0]
    else:
        valid = vesselness.ravel()
    if valid.size == 0 or valid.max() <= 0:
        return vesselness, np.zeros(img.shape, dtype=np.uint8)
    thresh = np.percentile(valid[valid > 0], threshold_percentile) if np.any(valid > 0) else 1.0
    mask = (vesselness >= thresh).astype(np.uint8)
    if visible_mask is not None:
        mask = mask * visible_mask.astype(np.uint8)
    return vesselness, mask


def tophat_vessel_mask(img: np.ndarray, kernel_size: int = 15, threshold_percentile: float = 97.0,
                        visible_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Top-hat morfológico (black-hat, ya que los vasos son oscuros):
    realza estructuras finas oscuras más chicas que `kernel_size` sobre un
    fondo suave -- alternativa clásica más barata que Frangi (un solo
    kernel, no un banco de escalas)."""
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    if visible_mask is not None:
        blackhat = blackhat * visible_mask.astype(np.float32)
        valid = blackhat[visible_mask > 0]
    else:
        valid = blackhat.ravel()
    if valid.size == 0 or valid.max() <= 0:
        return blackhat, np.zeros(img.shape, dtype=np.uint8)
    thresh = np.percentile(valid[valid > 0], threshold_percentile) if np.any(valid > 0) else 1.0
    mask = (blackhat >= thresh).astype(np.uint8)
    if visible_mask is not None:
        mask = mask * visible_mask.astype(np.uint8)
    return blackhat, mask


def visible_field_mask(fundus_rgb: np.ndarray, dark_threshold: int = 8) -> np.ndarray:
    """Máscara del campo visible circular del fundus: cualquier canal RGB
    por encima de `dark_threshold` cuenta como 'dentro de campo' -- excluye
    solo el negro puro fuera del círculo óptico, no la caída gradual de
    brillo (que es justamente el viñeteado que se está corrigiendo, no algo
    a recortar)."""
    return (fundus_rgb.max(axis=-1) > dark_threshold).astype(np.uint8)


def dice_against_gt(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")
