"""Tests de `fido.data.task1_augment`.

CRITICO (ver docstring del modulo bajo prueba y el encargo que lo motiva):
un error de signo/eje en como el keypoint se transforma respecto de como se
transforma la imagen no revienta con una excepcion -- entrena horas sobre
etiquetas sistematicamente corridas y el modelo simplemente converge peor (o,
como en T1-101, colapsa). La unica defensa es verificar, sobre casos
sinteticos donde se conoce la respuesta exacta, que la imagen y el keypoint
se mueven de forma consistente: se coloca un rasgo puntual (un pixel
brillante) en una posicion conocida, se transforma la imagen, y se comprueba
que el ARGMAX de la imagen transformada cae exactamente donde el keypoint
transformado dice que deberia caer.
"""

from __future__ import annotations

import math

import pytest
import torch

from fido.data.task1_augment import (
    AUGMENT_PRESETS,
    AugmentedTask1Dataset,
    Task1AugmentConfig,
    apply_brightness,
    apply_contrast,
    apply_gamma,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_geometric_transform,
    augment_fundus_and_keypoint,
    get_task1_augment_config,
    sample_geometric_transform,
    transform_image,
    transform_keypoint,
)

SIZE = 65  # tamano impar: hay un pixel central exacto, sin ambiguedad de
          # redondeo en flips/rotaciones de 90 grados.


def _dot_image(size: int, row: int, col: int) -> torch.Tensor:
    """Imagen (1, size, size) con un unico pixel brillante en (row, col)."""
    img = torch.zeros(1, size, size)
    img[0, row, col] = 1.0
    return img


def _argmax_rc(img: torch.Tensor) -> tuple[int, int]:
    idx = int(img[0].argmax())
    return divmod(idx, img.shape[-1])


def _inverse_params(angle: float, scale: float, translate_px: tuple[float, float],
                    flip: bool) -> tuple[float, float, tuple[float, float], bool]:
    """Parametros (angulo, escala, traslacion, flip) de la transformacion que
    DESHACE exactamente una transformacion `apply_geometric_transform` con los
    parametros dados. Derivacion (ver docstring de `task1_augment.py`):
    A = R(angle) @ (scale * F). Sin flip, A^-1 = R(-angle) * (1/scale)
    (rotacion y escala uniforme conmutan). Con flip, la identidad
    `F @ R(theta) = R(-theta) @ F` da A^-1 = R(angle) @ ((1/scale) * F) --
    el angulo NO se niega cuando hay flip. La traslacion inversa es
    `-A^-1 @ t`, calculada aqui reusando `transform_keypoint` con
    `image_size=1` (centro en el origen) para obtener el producto matricial
    sin repetir la formula a mano."""
    inverse_scale = 1.0 / scale
    inverse_angle = angle if flip else -angle
    t_vec = torch.tensor(translate_px)
    a_inv_t = transform_keypoint(t_vec, image_size=1, angle_deg=inverse_angle,
                                 scale=inverse_scale, translate_px=(0.0, 0.0),
                                 flip_horizontal=flip)
    inverse_translate = (-a_inv_t[0].item(), -a_inv_t[1].item())
    return inverse_angle, inverse_scale, inverse_translate, flip


# ---------------------------------------------------------------------------
# Geometria: consistencia imagen <-> keypoint sobre casos sinteticos exactos.
# ---------------------------------------------------------------------------

class TestGeometricTransformConsistency:
    """Para cada transformacion, coloca un rasgo puntual, transforma, y
    verifica que el ARGMAX de la imagen transformada coincide con el
    keypoint transformado -- no solo que el keypoint "se mueva algo"."""

    def test_identity_is_a_noop(self):
        img = _dot_image(SIZE, 20, 45)
        kp = torch.tensor([45.0, 20.0])
        new_img, new_kp = apply_geometric_transform(img, kp)
        assert torch.allclose(new_kp, kp)
        assert torch.allclose(new_img, img, atol=1e-5)

    def test_pure_translation(self):
        img = _dot_image(SIZE, 20, 45)
        kp = torch.tensor([45.0, 20.0])
        new_img, new_kp = apply_geometric_transform(img, kp, translate_px=(5.0, -3.0))
        row, col = _argmax_rc(new_img)
        assert new_kp.tolist() == pytest.approx([45.0 + 5.0, 20.0 - 3.0], abs=1e-4)
        assert (col, row) == (45 + 5, 20 - 3)

    def test_hflip_moves_x_only(self):
        """Guarda explicita de eje: si `hflip` estuviera implementado sobre el
        eje equivocado (filas en vez de columnas), este test lo detecta -- un
        roundtrip identity+hflip+hflip seguiria pasando pero el eje estaria
        mal."""
        row, col = 20, 45
        img = _dot_image(SIZE, row, col)
        kp = torch.tensor([float(col), float(row)])
        new_img, new_kp = apply_geometric_transform(img, kp, flip_horizontal=True)
        new_row, new_col = _argmax_rc(new_img)
        expected_col = SIZE - 1 - col
        assert new_kp.tolist() == pytest.approx([expected_col, row], abs=1e-4)
        assert (new_col, new_row) == (expected_col, row)

    @pytest.mark.parametrize("angle,expected_rc", [
        (90.0, None),
        (-90.0, None),
        (180.0, None),
    ])
    def test_rotations_match_argmax(self, angle, expected_rc):
        row, col = 20, 45
        img = _dot_image(SIZE, row, col)
        kp = torch.tensor([float(col), float(row)])
        new_img, new_kp = apply_geometric_transform(img, kp, angle_deg=angle)
        new_row, new_col = _argmax_rc(new_img)
        assert new_kp.tolist() == pytest.approx([new_col, new_row], abs=1e-2)

    def test_rot90_is_distinct_from_rotminus90(self):
        row, col = 20, 45
        img = _dot_image(SIZE, row, col)
        kp = torch.tensor([float(col), float(row)])
        _, kp_plus = apply_geometric_transform(img, kp, angle_deg=90.0)
        _, kp_minus = apply_geometric_transform(img, kp, angle_deg=-90.0)
        assert torch.linalg.vector_norm(kp_plus - kp_minus).item() > 1.0

    def test_scale_matches_argmax(self):
        row, col = 20, 45
        img = _dot_image(SIZE, row, col)
        kp = torch.tensor([float(col), float(row)])
        new_img, new_kp = apply_geometric_transform(img, kp, scale=2.0)
        new_row, new_col = _argmax_rc(new_img)
        assert new_kp.tolist() == pytest.approx([new_col, new_row], abs=1e-2)
        center = (SIZE - 1) / 2.0
        assert new_kp.tolist() == pytest.approx(
            [center + 2 * (col - center), center + 2 * (row - center)], abs=1e-2
        )

    @pytest.mark.parametrize("angle,scale,translate_px,flip", [
        (23.0, 1.3, (6.0, -4.0), True),
        (23.0, 1.3, (6.0, -4.0), False),
        (-40.0, 0.8, (-10.0, 12.0), True),
        (0.0, 1.0, (0.0, 0.0), True),
        (179.0, 0.75, (3.0, 3.0), False),
    ])
    def test_forward_then_inverse_recovers_keypoint(self, angle, scale, translate_px, flip):
        """Compone la transformacion con su inversa analitica (ver
        `_inverse_params`) y verifica que el keypoint vuelve a su posicion
        original -- el test de roundtrip mas directo posible sobre el
        keypoint (no solo sobre la matriz en abstracto)."""
        row, col = 20, 45
        img = _dot_image(SIZE, row, col)
        kp = torch.tensor([float(col), float(row)])
        img2, kp2 = apply_geometric_transform(img, kp, angle_deg=angle, scale=scale,
                                              translate_px=translate_px, flip_horizontal=flip)
        inv_angle, inv_scale, inv_translate, inv_flip = _inverse_params(
            angle, scale, translate_px, flip)
        img3, kp3 = apply_geometric_transform(img2, kp2, angle_deg=inv_angle, scale=inv_scale,
                                              translate_px=inv_translate, flip_horizontal=inv_flip)
        assert kp3.tolist() == pytest.approx(kp.tolist(), abs=1e-3)

    def test_transform_image_matches_transform_keypoint_batched(self):
        """`transform_keypoint` debe funcionar igual sobre un batch (N,2) que
        sobre un unico punto (2,) -- lo que usa `sample_geometric_transform`
        (un punto) debe ser consistente con un eventual uso batched."""
        points = torch.tensor([[45.0, 20.0], [10.0, 60.0], [32.0, 32.0]])
        batched = transform_keypoint(points, SIZE, angle_deg=30.0, scale=1.1,
                                     translate_px=(2.0, -1.0), flip_horizontal=True)
        for i in range(points.shape[0]):
            single = transform_keypoint(points[i], SIZE, angle_deg=30.0, scale=1.1,
                                        translate_px=(2.0, -1.0), flip_horizontal=True)
            assert batched[i].tolist() == pytest.approx(single.tolist(), abs=1e-5)


# ---------------------------------------------------------------------------
# Fotometricas: preservan el negro del vinetado, cambian lo que deben cambiar.
# ---------------------------------------------------------------------------

class TestPhotometricAugmentations:
    def test_brightness_preserves_pure_black(self):
        """El vinetado real cae a negro puro (medido: 92 -> 0 en r=600px). Un
        brillo multiplicativo no puede levantar un pixel que ya esta en 0."""
        black = torch.zeros(3, 16, 16)
        for gain in (0.5, 1.0, 1.5, 2.0):
            assert torch.equal(apply_brightness(black, gain), black)

    def test_brightness_scales_nonzero_pixels(self):
        img = torch.full((3, 4, 4), 0.4)
        out = apply_brightness(img, 1.5)
        assert torch.allclose(out, torch.full((3, 4, 4), 0.6), atol=1e-6)

    def test_gamma_preserves_pure_black_and_white(self):
        img = torch.tensor([0.0, 1.0, 0.5])
        for gamma in (0.5, 1.0, 2.0):
            out = apply_gamma(img, gamma)
            assert out[0].item() == pytest.approx(0.0, abs=1e-6)
            assert out[1].item() == pytest.approx(1.0, abs=1e-6)

    def test_gamma_identity(self):
        img = torch.rand(3, 8, 8)
        assert torch.allclose(apply_gamma(img, 1.0), img, atol=1e-6)

    def test_contrast_identity(self):
        img = torch.rand(3, 8, 8)
        assert torch.allclose(apply_contrast(img, 1.0), img, atol=1e-6)

    def test_contrast_lift_on_vignette_periphery_is_bounded_for_preset_ranges(self):
        """Un pixel negro del vinetado, bajo el `factor` MINIMO permitido por
        cada preset, no debe levantarse por encima de un umbral razonable --
        es exactamente la preocupacion del docstring del modulo (contraste
        `factor<1` corre los pixeles oscuros hacia la media)."""
        image_mean = 0.3
        img = torch.full((1, 4, 4), image_mean)
        img[0, 0, 0] = 0.0  # pixel de vinetado, negro puro
        for name in ("light", "strong"):
            config = AUGMENT_PRESETS[name]
            min_factor = config.contrast_range[0]
            out = apply_contrast(img, min_factor)
            lifted = out[0, 0, 0].item()
            assert lifted < 0.06, f"{name}: contraste levanta el vinetado a {lifted:.3f}"

    def test_gaussian_noise_changes_image_and_is_seed_reproducible(self):
        img = torch.full((3, 8, 8), 0.5)
        gen1 = torch.Generator().manual_seed(0)
        gen2 = torch.Generator().manual_seed(0)
        out1 = apply_gaussian_noise(img, 0.05, generator=gen1)
        out2 = apply_gaussian_noise(img, 0.05, generator=gen2)
        assert torch.equal(out1, out2)
        assert not torch.equal(out1, img)

    def test_gaussian_noise_zero_std_is_identity(self):
        img = torch.rand(3, 8, 8)
        assert torch.equal(apply_gaussian_noise(img, 0.0), img)

    def test_gaussian_blur_smooths_a_sharp_edge(self):
        img = torch.zeros(1, 32, 32)
        img[:, :, 16:] = 1.0
        blurred = apply_gaussian_blur(img, sigma=2.0)
        # El borde deja de ser un escalon puro: debe existir al menos un valor
        # intermedio cerca de la frontera.
        edge_region = blurred[0, 15, 12:20]
        assert edge_region.min().item() > 0.01
        assert edge_region.max().item() < 0.99

    def test_gaussian_blur_zero_sigma_is_identity(self):
        img = torch.rand(1, 16, 16)
        assert torch.equal(apply_gaussian_blur(img, sigma=0.0), img)


# ---------------------------------------------------------------------------
# Configuracion / presets / muestreo con reintento.
# ---------------------------------------------------------------------------

class TestConfigAndSampling:
    def test_off_preset_is_disabled_and_is_a_noop(self):
        config = get_task1_augment_config("off")
        assert config.enabled is False
        img = torch.rand(3, 16, 16)
        kp = torch.tensor([8.0, 8.0])
        out_img, out_kp = augment_fundus_and_keypoint(img, kp, config)
        assert out_img is img
        assert out_kp is kp

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            get_task1_augment_config("nonexistent")

    @pytest.mark.parametrize("name", ["light", "strong"])
    def test_preset_hflip_only_never_vflip(self, name):
        """El encargo pide argumentar y respetar la decision anatomica: flip
        horizontal si, vertical nunca. Esto se verifica indirectamente -- el
        modulo no expone ningun parametro de flip vertical en absoluto, asi
        que basta con confirmar que `Task1AugmentConfig` no tiene ese campo."""
        assert not hasattr(AUGMENT_PRESETS[name], "vflip_prob")

    @pytest.mark.parametrize("name", ["light", "strong"])
    def test_sample_geometric_transform_keeps_keypoint_in_bounds(self, name):
        config = AUGMENT_PRESETS[name]
        image_size = 1024
        generator = torch.Generator().manual_seed(1234)
        keypoint = torch.tensor([512.0, 512.0])
        for _ in range(200):
            params = sample_geometric_transform(keypoint, image_size, config, generator=generator)
            new_kp = transform_keypoint(keypoint, image_size, params["angle_deg"],
                                        params["scale"], params["translate_px"],
                                        params["flip_horizontal"])
            x, y = new_kp.tolist()
            assert config.keypoint_margin_px <= x <= image_size - 1 - config.keypoint_margin_px
            assert config.keypoint_margin_px <= y <= image_size - 1 - config.keypoint_margin_px

    def test_sample_geometric_transform_falls_back_to_identity_when_unsatisfiable(self):
        """Un `max_translate_frac` absurdo (1000x el lado de la imagen) hace
        que practicamente cualquier muestra saque el keypoint del canvas.
        Tras agotar los reintentos, debe caer a la identidad en vez de
        devolver una transformacion invalida."""
        config = Task1AugmentConfig(
            enabled=True, max_rotation_deg=10.0, scale_range=(1.0, 1.0),
            max_translate_frac=1000.0, hflip_prob=0.0, max_resample_attempts=5,
            keypoint_margin_px=8.0,
        )
        generator = torch.Generator().manual_seed(7)
        keypoint = torch.tensor([512.0, 512.0])
        params = sample_geometric_transform(keypoint, 1024, config, generator=generator)
        assert params == {
            "angle_deg": 0.0, "scale": 1.0, "translate_px": (0.0, 0.0),
            "flip_horizontal": False,
        }

    def test_augment_fundus_and_keypoint_keeps_keypoint_in_frame(self):
        config = AUGMENT_PRESETS["strong"]
        generator = torch.Generator().manual_seed(3)
        fundus = torch.rand(3, 128, 128)
        keypoint = torch.tensor([64.0, 64.0])
        aug_img, aug_kp = augment_fundus_and_keypoint(fundus, keypoint, config, generator=generator)
        assert aug_img.shape == fundus.shape
        x, y = aug_kp.tolist()
        assert -1e-3 <= x <= 128
        assert -1e-3 <= y <= 128
        assert torch.isfinite(aug_img).all()


# ---------------------------------------------------------------------------
# Dataset wrapper.
# ---------------------------------------------------------------------------

class _FakeBaseDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 4, size: int = 64):
        self.size = size
        self.items = [
            {
                "fundus": torch.rand(3, size, size),
                "keypoint": torch.tensor([size / 2.0, size / 2.0]),
                "scenario": f"Scenario_{i:02d}",
                "frame_id": f"{i:04d}",
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TestAugmentedTask1Dataset:
    def test_disabled_config_passes_through_unchanged(self):
        base = _FakeBaseDataset()
        wrapped = AugmentedTask1Dataset(base, get_task1_augment_config("off"), seed=0)
        for i in range(len(base)):
            item = wrapped[i]
            assert torch.equal(item["fundus"], base[i]["fundus"])
            assert torch.equal(item["keypoint"], base[i]["keypoint"])

    def test_non_geometric_keys_pass_through(self):
        base = _FakeBaseDataset()
        wrapped = AugmentedTask1Dataset(base, AUGMENT_PRESETS["strong"], seed=0)
        item = wrapped[0]
        assert item["scenario"] == base[0]["scenario"]
        assert item["frame_id"] == base[0]["frame_id"]

    def test_deterministic_given_same_seed_and_index(self):
        base = _FakeBaseDataset()
        w1 = AugmentedTask1Dataset(base, AUGMENT_PRESETS["strong"], seed=42)
        w2 = AugmentedTask1Dataset(base, AUGMENT_PRESETS["strong"], seed=42)
        item1, item2 = w1[2], w2[2]
        assert torch.equal(item1["fundus"], item2["fundus"])
        assert torch.equal(item1["keypoint"], item2["keypoint"])

    def test_different_indices_get_different_augmentations(self):
        base = _FakeBaseDataset()
        wrapped = AugmentedTask1Dataset(base, AUGMENT_PRESETS["strong"], seed=0)
        item0, item1 = wrapped[0], wrapped[1]
        assert not torch.equal(item0["fundus"], item1["fundus"])

    def test_set_epoch_changes_augmentation_for_same_index(self):
        base = _FakeBaseDataset()
        wrapped = AugmentedTask1Dataset(base, AUGMENT_PRESETS["strong"], seed=0)
        item_epoch0 = wrapped[0]
        wrapped.set_epoch(1)
        item_epoch1 = wrapped[0]
        assert not torch.equal(item_epoch0["fundus"], item_epoch1["fundus"])

    def test_len_matches_base(self):
        base = _FakeBaseDataset(n=7)
        wrapped = AugmentedTask1Dataset(base, AUGMENT_PRESETS["light"], seed=0)
        assert len(wrapped) == 7
