"""Test-time augmentation (TTA) para el keypoint de Task 1 — fusion en espacio
de HEATMAP, un solo decode al final.

Por que fusionar heatmaps y no coordenadas: promediar coordenadas descarta la
forma del heatmap de cada vista (que tan seguro estaba el modelo, si hubo un
modo espurio) y trata cada vista como un punto ciego. Promediar los LOGITS del
heatmap -- ya des-transformados a la grilla canonica -- preserva esa
informacion y permite decodificar una sola vez con el mismo decodificador
sub-pixel (`fido.heatmap_decode`) que usa el modelo sin TTA, asi que el unico
efecto que se mide es el de la augmentacion, no el del decoder.

CRITICO (por que este modulo tiene tests sinteticos exhaustivos en
`fido/tests/test_tta_keypoint.py`): cada transformacion geometrica aplicada a
la imagen de entrada tiene que deshacerse EXACTAMENTE sobre el heatmap de
salida antes de promediar. Un error de signo en un flip, o confundir fila/
columna en una rotacion de 90 grados, no revienta con una excepcion -- corre,
converge a un numero, y ese numero es peor que no hacer TTA. La unica forma de
detectarlo es verificar que cada transformacion, compuesta con su inversa,
recupera la coordenada original.

Familias de transformaciones:
  - Discretas (flip horizontal/vertical, rotaciones de 90/180/270): se
    implementan con `torch.flip`/`torch.rot90`, permutaciones EXACTAS de la
    grilla -- no hay interpolacion de por medio, el roundtrip es exacto salvo
    error de punto flotante (~1e-5 px).
  - Continua (zoom multi-escala): se implementa con una rejilla afin
    (`F.affine_grid` + `F.grid_sample`) sobre coordenadas normalizadas
    [-1, 1], que son independientes de la resolucion -- la MISMA matriz sirve
    para transformar la imagen (1024x1024) y para des-transformar el heatmap
    (p.ej. 64x64) sin tener que convertir manualmente entre escalas. El
    roundtrip aqui tiene error de interpolacion bilineal, no es exactamente
    cero, pero debe ser sub-pixel.

Fusion en LOGITS (no en probabilidades sigmoid): promediar logits equivale a
la media geometrica de las probabilidades (producto de odds), que preserva
picos mas afilados que promediar probabilidades directamente -- relevante
porque el decodificador sub-pixel (`soft_argmax_2d`) espera logits de rango
amplio, no probabilidades acotadas en [0,1]. El padding introducido por el
zoom usa `padding_mode="border"`: replica el logit del borde en vez de
insertar un logit=0 (que seria una probabilidad ~0.5, una senal de "aqui hay
algo" espuria en medio de lo que casi siempre es fondo muy negativo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from fido.heatmap_decode import decode_heatmap

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class KeypointTTATransform:
    """Un par (transformar imagen, des-transformar heatmap) que se cancelan
    exactamente entre si cuando se componen: `untransform_heatmap` deshace
    sobre el heatmap lo que `transform_image` le hizo a la imagen."""

    name: str
    transform_image: TensorTransform
    untransform_heatmap: TensorTransform


def _identity_transform() -> KeypointTTATransform:
    return KeypointTTATransform("identity", lambda x: x, lambda x: x)


def _flip_transform(name: str, dim: int) -> KeypointTTATransform:
    """Flip a lo largo de `dim` (-1 = columnas/horizontal, -2 = filas/vertical).
    Es su propia inversa: aplicarlo dos veces es la identidad."""

    def apply(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(dim,))

    return KeypointTTATransform(name, apply, apply)


def _rotate90_transform(k: int) -> KeypointTTATransform:
    """Rotacion de `k * 90` grados (convencion `torch.rot90`, sentido
    antihorario, sobre las dos ultimas dimensiones). La inversa es rotar
    `-k` (equivalente a `4 - k` modulo 4)."""
    if k not in (1, 2, 3):
        raise ValueError("k must be 1, 2 or 3 (90/180/270 degrees)")

    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=k, dims=(-2, -1))

    def inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=-k, dims=(-2, -1))

    return KeypointTTATransform(f"rot{90 * k}", forward, inverse)


def _affine_resample(x: torch.Tensor, diagonal: float,
                      padding_mode: str) -> torch.Tensor:
    """Re-muestrea `x` (B,C,H,W) con una matriz afin diagonal (sin traslacion
    ni rotacion) en coordenadas normalizadas [-1, 1]. `F.affine_grid` mapea
    cada posicion de SALIDA a una posicion de ENTRADA via `theta`, asi que un
    `diagonal` menor que 1 hace zoom-in (solo se muestrea la region central de
    la entrada) y mayor que 1 hace zoom-out (se muestrea mas alla del borde,
    de ahi `padding_mode`)."""
    batch = x.shape[0]
    theta = x.new_tensor([[diagonal, 0.0, 0.0], [0.0, diagonal, 0.0]])
    theta = theta.unsqueeze(0).expand(batch, -1, -1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode,
                          align_corners=False)


def _zoom_transform(scale: float, image_padding_mode: str = "reflection",
                     heatmap_padding_mode: str = "border") -> KeypointTTATransform:
    """`scale > 1` acerca (zoom-in): la imagen transformada muestra solo la
    region central de la original, ampliada a la misma resolucion de lienzo.
    `scale < 1` aleja (zoom-out): dentro del mismo lienzo se ve mas contexto
    del que la imagen realmente tiene, rellenando el resto.

    Para deshacer sobre el heatmap se usa la matriz RECIPROCA: si la imagen se
    muestreo con diagonal `1/scale` (para lograr el zoom `scale`), el heatmap
    resultante vive en esa misma grilla deformada, y se vuelve a la grilla
    canonica muestreando con diagonal `scale`. Verificado en
    `test_tta_keypoint.py::test_zoom_roundtrip_recovers_coordinate`.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    inverse_scale = 1.0 / scale

    def forward(image: torch.Tensor) -> torch.Tensor:
        return _affine_resample(image, inverse_scale, image_padding_mode)

    def inverse(heatmap: torch.Tensor) -> torch.Tensor:
        return _affine_resample(heatmap, scale, heatmap_padding_mode)

    return KeypointTTATransform(f"zoom_{scale:.2f}", forward, inverse)


def default_flip_rotation_transforms() -> list[KeypointTTATransform]:
    """Las 6 vistas discretas: identidad + 2 flips + 3 rotaciones de 90. No
    incluye las combinaciones flip+rotacion (el grupo diedral D4 completo
    tiene 8 elementos) a proposito -- 6 vistas ya cubren horizontal, vertical
    y las 4 orientaciones cardinales, y cada vista adicional cuesta un forward
    pass completo (ver medicion de coste en INFORME.md)."""
    return [
        _identity_transform(),
        _flip_transform("hflip", dim=-1),
        _flip_transform("vflip", dim=-2),
        _rotate90_transform(1),
        _rotate90_transform(2),
        _rotate90_transform(3),
    ]


def default_scale_transforms(scales: tuple[float, ...] = (0.9, 1.1)
                              ) -> list[KeypointTTATransform]:
    """Vistas de zoom alrededor de la escala nativa 1.0 (que ya cubre la vista
    `identity` de `default_flip_rotation_transforms`, por eso este helper NO
    incluye una vista `scale=1.0` -- se compone con la familia de flips/
    rotaciones, que ya aporta la vista sin deformar)."""
    return [_zoom_transform(scale) for scale in scales]


def tta_forward_logits(model: torch.nn.Module, fundus: torch.Tensor,
                       transforms: list[KeypointTTATransform],
                       fundus_size: int | None = None) -> torch.Tensor:
    """Corre `model` sobre cada vista transformada de `fundus`, deshace la
    transformacion geometrica sobre `heatmap_logits` y devuelve el promedio
    en la grilla canonica (la del heatmap sin transformar). No decodifica --
    eso lo hace `tta_predict_keypoint`, para que quien solo quiera el heatmap
    fusionado (p.ej. para comparar decoders) no pague un decode de mas."""
    if not transforms:
        raise ValueError("transforms must be a non-empty list")
    image_size = fundus_size or fundus.shape[-1]
    accumulated = None
    for view in transforms:
        transformed = view.transform_image(fundus)
        with torch.no_grad():
            output = model(transformed, fundus_size=image_size)
        heatmap = view.untransform_heatmap(output["heatmap_logits"])
        accumulated = heatmap if accumulated is None else accumulated + heatmap
    return accumulated / len(transforms)


def tta_predict_keypoint(model: torch.nn.Module, fundus: torch.Tensor,
                         transforms: list[KeypointTTATransform],
                         fundus_size: int | None = None,
                         decode_mode: str = "global",
                         temperature: float | None = None,
                         window: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """TTA de punta a punta: fusiona heatmaps (ver `tta_forward_logits`) y
    decodifica UNA vez. Devuelve `(keypoint_xy, mean_heatmap_logits)`, con
    `keypoint_xy` en pixeles reales del fundus -- misma convencion que
    `model.forward()["keypoint"]`, para poder sustituir uno por otro sin
    tocar el resto del pipeline de evaluacion."""
    image_size = fundus_size or fundus.shape[-1]
    mean_logits = tta_forward_logits(model, fundus, transforms, image_size)
    resolved_temperature = (temperature if temperature is not None
                            else float(getattr(model, "heatmap_temperature", 1.0)))
    coords_heatmap = decode_heatmap(mean_logits, mode=decode_mode,
                                    temperature=resolved_temperature, window=window)
    stride = image_size / mean_logits.shape[-1]
    keypoint_xy = coords_heatmap[:, 0, :] * stride
    return keypoint_xy, mean_logits


def ensemble_forward_logits(models: list[torch.nn.Module], fundus: torch.Tensor,
                            transforms: list[KeypointTTATransform],
                            fundus_size: int | None = None) -> torch.Tensor:
    """Promedia `tta_forward_logits` (TTA ya incluido si `transforms` tiene
    mas de una vista) sobre varios checkpoints. Todos los modelos deben
    compartir el mismo stride/tamano de heatmap para el mismo `fundus_size`
    (cierto para checkpoints del mismo `Task1KeypointModel`)."""
    if not models:
        raise ValueError("models must be a non-empty list")
    accumulated = None
    for model in models:
        logits = tta_forward_logits(model, fundus, transforms, fundus_size)
        accumulated = logits if accumulated is None else accumulated + logits
    return accumulated / len(models)


def ensemble_predict_keypoint(models: list[torch.nn.Module], fundus: torch.Tensor,
                              transforms: list[KeypointTTATransform],
                              fundus_size: int | None = None,
                              decode_mode: str = "global",
                              temperature: float = 1.0,
                              window: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = fundus_size or fundus.shape[-1]
    mean_logits = ensemble_forward_logits(models, fundus, transforms, image_size)
    coords_heatmap = decode_heatmap(mean_logits, mode=decode_mode,
                                    temperature=temperature, window=window)
    stride = image_size / mean_logits.shape[-1]
    keypoint_xy = coords_heatmap[:, 0, :] * stride
    return keypoint_xy, mean_logits
