"""Augmentacion geometrica y fotometrica para el keypoint de Task 1.

Motivacion (ver `experiments/103-t1-augmentation/PRE_REGISTRATION.md`): T1-101
midio que TTA con flips/rotaciones DESTRUYE el checkpoint de produccion (AUC
0.8697 -> 0.0061). Los 45 tests de `test_tta_keypoint.py` descartan un bug de
des-transformacion -- el roundtrip transformacion+inversa recupera la
coordenada a <1e-4 px. La explicacion que queda es que el modelo nunca vio,
durante el entrenamiento, ninguna imagen rotada/reflejada/reescalada: es
equivariante por accidente (si acaso) en la vecindad de la distribucion de
entrenamiento, no por diseno. `train_task1_keypoint.py` alimenta el fundus
crudo a `compute_loss` sin ninguna transformacion (confirmado leyendo el loop
de entrenamiento completo) -- CERO augmentacion hoy, geometrica o fotometrica.

Este modulo implementa esa augmentacion para atacar dos problemas a la vez:
fragilidad geometrica (motiva este peldano) y el domain shift de apariencia
entre escenarios (motiva la parte fotometrica, ver `ATTACK_LADDER.md`
brecha local-real de 0.10 AUC).

## Convencion geometrica (debe coincidir con `fido.tta_keypoint` y
`fido.models.task1_keypoint`)

- La imagen es un tensor `(C, H, W)` con `H == W` (fundus cuadrado, 1024 por
  defecto). `dim=-1` son columnas (eje x), `dim=-2` son filas (eje y).
- El keypoint es `(x, y)` en pixeles reales del fundus, mismo origen que usa
  `Task1Dataset` (`gt[:2]` del JSON de "Ground Truth" -> "Task 1") y
  `Task1KeypointModel.forward` (`keypoint = coords * stride`).
- Todas las transformaciones geometricas de este modulo se definen como una
  transformacion RIGIDA+escala del CONTENIDO de la imagen (un rasgo en el
  pixel `p_in` termina en `p_out = A @ (p_in - centro) + t + centro`). El
  KEYPOINT se transforma con exactamente esa misma formula -- es la unica
  forma de garantizar que la etiqueta sigue apuntando al mismo rasgo despues
  de transformar la imagen. La imagen, en cambio, se remuestrea con la
  transformacion INVERSA (`A^-1`, `-A^-1 @ t`) via `grid_sample`, porque
  `grid_sample`/`affine_grid` esperan el mapa de SALIDA -> ENTRADA (ver
  docstring de `fido.tta_keypoint._affine_resample`, mismo patron). Aplicar
  la formula equivocada a cualquiera de los dos (usar la inversa en el
  keypoint, o la directa en la imagen) es exactamente el bug de signo que
  este modulo esta disenado para evitar -- por eso `apply_geometric_transform`
  es la UNICA funcion que decide el signo, y todo lo demas la reusa.

## Decisiones de diseno (repasadas explicitamente por instrucciones del
encargo, no por defecto)

- **Flip horizontal SI, flip vertical NO.** La imagen es la vista del
  microscopio estereo sobre el ojo durante cirugia. Un flip horizontal
  convierte una vista de ojo derecho en una vista de ojo izquierdo
  (posicion del canal/instrumento especular) -- ambas son anatomias
  validas que de hecho aparecen en el dataset (los organizadores operan
  ambos ojos). Un flip VERTICAL invierte la quiralidad de la escena sin que
  exista ninguna pose fisica de camara/paciente que lo produzca: no es "mirar
  el mismo ojo desde otro angulo", es una anatomia imposible (equivalente a
  ver el instrumento entrando desde la direccion opuesta a como lo sujeta la
  mano del cirujano). Por eso NO se ofrece flip vertical, ni siquiera en la
  intensidad "strong".
- **Rotacion completa (hasta +-180 grados en `strong`) SI es fisica.** A
  diferencia del flip, rotar la imagen alrededor del eje optico corresponde a
  un giro real del microscopio/cabezal del paciente durante la cirugia --
  preserva la quiralidad (no espeja nada), solo cambia la orientacion "arriba"
  de la escena. Es la transformacion que TTA aplico y que colapso el modelo;
  entrenar con ella es la prueba directa de la hipotesis del encargo.
- **El vinetado (medido: 92 -> 0 de brillo entre el centro y r=600px) acota
  las augmentaciones fotometricas aditivas.** Un brillo ADITIVO
  (`img + delta`) levantaria uniformemente los pixeles casi negros de la
  periferia del vinetado por encima de cero -- una imagen que el simulador
  nunca produciria (el vinetado real llega a negro puro). Por eso el brillo
  aqui es MULTIPLICATIVO (`img * gain`) y el gamma es una potencia
  (`img ** gamma`): ambas formas fijan el punto `img=0` (`0 * gain == 0`,
  `0 ** gamma == 0`), preservando el negro del vinetado exactamente donde el
  simulador lo puso. El contraste (`(img - mean) * factor + mean`) SI corre
  el riesgo de levantar la periferia cuando `factor < 1` (acerca los pixeles
  oscuros hacia la media) -- se mantiene deliberadamente acotado
  (`0.85 <= factor <= 1.2` incluso en `strong`) para que ese efecto sea
  pequeno, no para eliminarlo del todo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


# ---------------------------------------------------------------------------
# Transformacion geometrica determinista (sin aleatoriedad) -- el nucleo que
# los tests verifican exhaustivamente.
# ---------------------------------------------------------------------------

def _rotation_matrix(angle_deg: float, dtype: torch.dtype) -> torch.Tensor:
    """Matriz de rotacion 2x2 en el plano (x, y) de la imagen. Angulo positivo
    gira el CONTENIDO en sentido antihorario en coordenadas matematicas
    estandar (x a la derecha, y hacia arriba); como aqui y crece hacia abajo
    (fila de imagen), el efecto visual es horario -- consistente en todo el
    modulo porque la MISMA matriz se usa para keypoint e imagen (una via
    directa, la otra via su inversa), asi que el sentido no afecta la
    correctitud del roundtrip, solo la convencion visual de "angulo positivo"."""
    theta = math.radians(angle_deg)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    return torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=dtype)


def _forward_affine(angle_deg: float, scale: float, flip_horizontal: bool,
                     dtype: torch.dtype) -> torch.Tensor:
    """Parte lineal 2x2 de la transformacion DIRECTA (aplicada al contenido /
    al keypoint): flip -> escala -> rotacion, en ese orden de composicion."""
    flip = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=dtype) if flip_horizontal \
        else torch.eye(2, dtype=dtype)
    rotation = _rotation_matrix(angle_deg, dtype)
    return rotation @ (scale * flip)


def transform_keypoint(
    keypoint_xy: torch.Tensor,
    image_size: int,
    angle_deg: float,
    scale: float,
    translate_px: tuple[float, float],
    flip_horizontal: bool,
) -> torch.Tensor:
    """Aplica la transformacion DIRECTA al keypoint `(..., 2)` en pixeles
    reales del fundus. `image_size` es el lado del fundus cuadrado."""
    dtype = keypoint_xy.dtype
    linear = _forward_affine(angle_deg, scale, flip_horizontal, dtype)
    center = (image_size - 1) / 2.0
    translation = keypoint_xy.new_tensor(translate_px)
    centered = keypoint_xy - center
    transformed = centered @ linear.T + translation
    return transformed + center


def transform_image(
    image: torch.Tensor,
    angle_deg: float,
    scale: float,
    translate_px: tuple[float, float],
    flip_horizontal: bool,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Remuestrea `image` (C, H, W) o (B, C, H, W) con la INVERSA de la
    transformacion directa de `transform_keypoint` -- ver docstring del
    modulo para por que el signo es el opuesto entre imagen y keypoint.

    `padding_mode="zeros"` (default): el borde extendido por rotacion/
    traslacion se rellena con negro puro, coherente con que el vinetado del
    fundus ya cae a negro cerca del borde (ver docstring del modulo) --
    "reflection" o "border" introducirian tejido/instrumento duplicado o
    estirado que el simulador nunca genera.
    """
    squeeze = image.dim() == 3
    x = image.unsqueeze(0) if squeeze else image
    batch, _, height, width = x.shape
    if height != width:
        raise ValueError("transform_image requires a square image")
    size = height
    dtype = x.dtype

    linear = _forward_affine(angle_deg, scale, flip_horizontal, dtype)
    inverse_linear = torch.linalg.inv(linear)
    translation = torch.tensor(translate_px, dtype=dtype)
    inverse_translation = -(inverse_linear @ translation)

    # theta de affine_grid mapea coordenadas normalizadas [-1,1] de SALIDA a
    # coordenadas normalizadas de ENTRADA. Con align_corners=False el mismo
    # factor de escala pixel<->normalizado (2/size) aplica por igual a la
    # parte lineal (es una razon, se cancela) y a la traslacion (no se
    # cancela, hay que convertirla explicitamente).
    theta = torch.zeros((1, 2, 3), dtype=dtype)
    theta[0, :, :2] = inverse_linear
    theta[0, :, 2] = inverse_translation * (2.0 / size)
    theta = theta.expand(batch, -1, -1)

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    warped = F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode,
                            align_corners=False)
    return warped.squeeze(0) if squeeze else warped


def apply_geometric_transform(
    image: torch.Tensor,
    keypoint_xy: torch.Tensor,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    translate_px: tuple[float, float] = (0.0, 0.0),
    flip_horizontal: bool = False,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica la MISMA transformacion geometrica a `image` (C,H,W) y a
    `keypoint_xy` (2,) de forma consistente. Punto de entrada unico para
    evitar que imagen y keypoint diverjan (ver docstring del modulo)."""
    image_size = image.shape[-1]
    new_keypoint = transform_keypoint(keypoint_xy, image_size, angle_deg, scale,
                                      translate_px, flip_horizontal)
    new_image = transform_image(image, angle_deg, scale, translate_px,
                                flip_horizontal, padding_mode=padding_mode)
    return new_image, new_keypoint


# ---------------------------------------------------------------------------
# Fotometricas -- no tocan el keypoint.
# ---------------------------------------------------------------------------

def apply_brightness(image: torch.Tensor, gain: float) -> torch.Tensor:
    """Multiplicativo: fija `img=0` (preserva negro del vinetado). `gain=1`
    es identidad."""
    return (image * gain).clamp(0.0, 1.0)


def apply_gamma(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Correccion gamma: `img ** gamma`. Fija `img=0` y `img=1`. `gamma=1`
    es identidad."""
    return image.clamp(0.0, 1.0).pow(gamma)


def apply_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Contraste clasico alrededor de la media global de la imagen. `factor=1`
    es identidad. Ver docstring del modulo: `factor < 1` levanta ligeramente
    la periferia del vinetado -- por eso los presets acotan `factor` cerca
    de 1 incluso en `strong`."""
    mean = image.mean()
    return ((image - mean) * factor + mean).clamp(0.0, 1.0)


def apply_gaussian_noise(image: torch.Tensor, std: float,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    if std <= 0:
        return image
    noise = torch.randn(image.shape, dtype=image.dtype, generator=generator)
    return (image + noise * std).clamp(0.0, 1.0)


def apply_gaussian_blur(image: torch.Tensor, sigma: float,
                        kernel_size: int | None = None) -> torch.Tensor:
    if sigma <= 0:
        return image
    if kernel_size is None:
        # Regla usual: kernel >= 6*sigma, impar. Acotado a >=3.
        kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    was_3d = image.dim() == 3
    x = image.unsqueeze(0) if was_3d else image
    blurred = gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blurred.squeeze(0) if was_3d else blurred


# ---------------------------------------------------------------------------
# Configuracion e intensidades.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task1AugmentConfig:
    """Rangos de muestreo. Todos los angulos en grados, traslaciones como
    fraccion del lado de la imagen, escala como factor multiplicativo."""

    enabled: bool = False
    max_rotation_deg: float = 0.0
    scale_range: tuple[float, float] = (1.0, 1.0)
    max_translate_frac: float = 0.0
    hflip_prob: float = 0.0
    brightness_gain_range: tuple[float, float] = (1.0, 1.0)
    gamma_range: tuple[float, float] = (1.0, 1.0)
    contrast_range: tuple[float, float] = (1.0, 1.0)
    max_noise_std: float = 0.0
    blur_prob: float = 0.0
    blur_sigma_range: tuple[float, float] = (0.0, 0.0)
    # Reintentos si el keypoint transformado cae fuera de la imagen (o muy
    # cerca del borde): mejor reintentar con nuevos parametros aleatorios que
    # entrenar contra una etiqueta invalida o fuera de canvas.
    max_resample_attempts: int = 8
    keypoint_margin_px: float = 8.0


AUGMENT_PRESETS: dict[str, Task1AugmentConfig] = {
    "off": Task1AugmentConfig(enabled=False),
    "light": Task1AugmentConfig(
        enabled=True,
        max_rotation_deg=15.0,
        scale_range=(0.9, 1.1),
        max_translate_frac=0.05,
        hflip_prob=0.5,
        brightness_gain_range=(0.9, 1.1),
        gamma_range=(0.9, 1.1),
        contrast_range=(0.95, 1.05),
        max_noise_std=0.02,
        blur_prob=0.2,
        blur_sigma_range=(0.5, 1.0),
    ),
    "strong": Task1AugmentConfig(
        enabled=True,
        max_rotation_deg=180.0,
        scale_range=(0.75, 1.3),
        max_translate_frac=0.15,
        hflip_prob=0.5,
        brightness_gain_range=(0.7, 1.3),
        gamma_range=(0.7, 1.4),
        contrast_range=(0.85, 1.2),
        max_noise_std=0.05,
        blur_prob=0.3,
        blur_sigma_range=(0.5, 2.0),
    ),
}


def get_task1_augment_config(name: str) -> Task1AugmentConfig:
    try:
        return AUGMENT_PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Task 1 augment intensity {name!r}; expected one of {sorted(AUGMENT_PRESETS)}"
        ) from exc


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    if low == high:
        return float(low)
    return float(torch.empty(1).uniform_(low, high, generator=generator).item())


def _sample_geometric_params(config: Task1AugmentConfig,
                             generator: torch.Generator | None) -> dict:
    angle = _uniform(-config.max_rotation_deg, config.max_rotation_deg, generator)
    scale = _uniform(*config.scale_range, generator)
    max_shift = config.max_translate_frac  # se multiplica por image_size en el llamador
    tx = _uniform(-max_shift, max_shift, generator)
    ty = _uniform(-max_shift, max_shift, generator)
    flip = bool(_uniform(0.0, 1.0, generator) < config.hflip_prob)
    return {"angle_deg": angle, "scale": scale, "translate_frac": (tx, ty), "flip_horizontal": flip}


def _keypoint_in_bounds(keypoint_xy: torch.Tensor, image_size: int, margin: float) -> bool:
    x, y = float(keypoint_xy[0]), float(keypoint_xy[1])
    return margin <= x <= (image_size - 1 - margin) and margin <= y <= (image_size - 1 - margin)


def sample_geometric_transform(
    keypoint_xy: torch.Tensor,
    image_size: int,
    config: Task1AugmentConfig,
    generator: torch.Generator | None = None,
) -> dict:
    """Muestrea parametros geometricos validos (el keypoint transformado cae
    dentro de la imagen con margen `keypoint_margin_px`), reintentando hasta
    `max_resample_attempts` veces. Si ninguno es valido, devuelve la
    identidad -- preferible a envenenar una etiqueta con un keypoint fuera
    de canvas."""
    for _ in range(max(1, config.max_resample_attempts)):
        params = _sample_geometric_params(config, generator)
        tx_px = params["translate_frac"][0] * image_size
        ty_px = params["translate_frac"][1] * image_size
        candidate_keypoint = transform_keypoint(
            keypoint_xy, image_size, params["angle_deg"], params["scale"],
            (tx_px, ty_px), params["flip_horizontal"],
        )
        if _keypoint_in_bounds(candidate_keypoint, image_size, config.keypoint_margin_px):
            return {
                "angle_deg": params["angle_deg"],
                "scale": params["scale"],
                "translate_px": (tx_px, ty_px),
                "flip_horizontal": params["flip_horizontal"],
            }
    return {"angle_deg": 0.0, "scale": 1.0, "translate_px": (0.0, 0.0), "flip_horizontal": False}


def augment_fundus_and_keypoint(
    fundus: torch.Tensor,
    keypoint_xy: torch.Tensor,
    config: Task1AugmentConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Punto de entrada de un solo ejemplo `(fundus, keypoint)` -- lo que usa
    `AugmentedTask1Dataset.__getitem__`. Geometrica primero (usa la imagen sin
    tocar para no acumular error de interpolacion de un blur/ruido previo
    dentro del muestreo del grid), fotometrica despues."""
    if not config.enabled:
        return fundus, keypoint_xy

    image_size = fundus.shape[-1]
    geo_params = sample_geometric_transform(keypoint_xy, image_size, config, generator)
    image, keypoint = apply_geometric_transform(
        fundus, keypoint_xy,
        angle_deg=geo_params["angle_deg"], scale=geo_params["scale"],
        translate_px=geo_params["translate_px"], flip_horizontal=geo_params["flip_horizontal"],
    )

    image = apply_brightness(image, _uniform(*config.brightness_gain_range, generator))
    image = apply_gamma(image, _uniform(*config.gamma_range, generator))
    image = apply_contrast(image, _uniform(*config.contrast_range, generator))
    image = apply_gaussian_noise(image, _uniform(0.0, config.max_noise_std, generator), generator=generator)
    if config.blur_prob > 0 and _uniform(0.0, 1.0, generator) < config.blur_prob:
        image = apply_gaussian_blur(image, _uniform(*config.blur_sigma_range, generator))

    return image, keypoint


class AugmentedTask1Dataset(torch.utils.data.Dataset):
    """Envuelve un `Task1Dataset` (o un `Subset` sobre el) sin tocarlo: aplica
    augmentacion sobre `fundus`/`keypoint` al vuelo en `__getitem__`, deja
    todas las demas claves (`scenario`, `frame_id`, `distance`, ...) intactas.

    Determinismo bajo `num_workers > 0`: cada `__getitem__` deriva su propio
    `torch.Generator` de `(seed, idx, epoch_salt)` en vez de usar el RNG
    global -- evita que dos workers con la misma copia fork-eada del proceso
    produzcan la misma "aleatoriedad" para indices distintos, y hace que la
    corrida sea reproducible dado el mismo `seed`."""

    def __init__(self, base_dataset: torch.utils.data.Dataset, config: Task1AugmentConfig,
                 seed: int = 0):
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Opcional: llamar al inicio de cada epoca para que la augmentacion
        no repita exactamente la misma vista en cada pasada del dataset.

        CAVEAT conocido con `DataLoader(persistent_workers=True, num_workers>0)`:
        los procesos worker reciben una COPIA del dataset al arrancar y la
        mantienen viva entre epocas -- una mutacion de `self._epoch` en el
        proceso principal NO se propaga a esas copias. Con esa combinacion
        (la que usa `train_task1_keypoint.py` por defecto en el pod,
        `--num-workers 8`), el efecto practico es que cada indice recibe UNA
        vista aumentada fija, reusada en todas las epocas -- sigue siendo
        augmentacion real (mejor que ninguna) pero sin resampleo por epoca.
        Confirmado que `set_epoch` si tiene efecto con `num_workers=0`
        (el caso del smoke test en CPU). Arreglarlo de raiz requeriria
        `worker_init_fn` o memoria compartida entre procesos -- fuera de
        alcance de este peldano, documentado en
        `experiments/103-t1-augmentation/PRE_REGISTRATION.md`."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base_dataset[idx])
        if not self.config.enabled:
            return item
        generator = torch.Generator()
        generator.manual_seed((self.seed * 1_000_003 + idx * 97 + self._epoch) % (2 ** 31 - 1))
        fundus, keypoint = augment_fundus_and_keypoint(
            item["fundus"], item["keypoint"], self.config, generator=generator
        )
        item["fundus"] = fundus
        item["keypoint"] = keypoint
        return item
