import pytest
import torch

from fido.heatmap_decode import decode_heatmap, gaussian_heatmap_target
from fido.models.task1_keypoint import Task1KeypointModel
from fido.tta_keypoint import (
    default_flip_rotation_transforms,
    default_scale_transforms,
    ensemble_predict_keypoint,
    tta_forward_logits,
    tta_predict_keypoint,
)

HEATMAP_SIZE = 64
SIGMA = 2.0


def _synthetic_logits(center_xy: tuple[float, float], size: int = HEATMAP_SIZE,
                      sigma: float = SIGMA) -> torch.Tensor:
    """Heatmap sintetico de un solo pico gaussiano en `center_xy` (coordenadas
    de grilla del heatmap), como logits (misma convencion que
    `Task1KeypointModel.heatmap_head`: BCE con logits crudos, sin sigmoid)."""
    center = torch.tensor([[list(center_xy)]], dtype=torch.float32)
    probability = gaussian_heatmap_target(center, size, size, sigma=sigma)
    return torch.logit(probability.clamp(1e-6, 1 - 1e-6))


def _decode_xy(logits: torch.Tensor) -> torch.Tensor:
    return decode_heatmap(logits, mode="global")[0, 0]


# ---------------------------------------------------------------------------
# CRITICO: cada transformacion + su inversa deben recuperar la coordenada
# original a ~0 px. Un error de signo/eje aqui degrada el TTA en silencio
# (converge a un numero, sin excepcion) -- por eso cada transformacion tiene
# su propio test aislado, no solo un test de "la lista completa funciona".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("center_xy", [
    (32.0, 32.0), (10.25, 50.75), (5.0, 58.0), (58.0, 5.0), (20.3, 20.3),
])
class TestDiscreteTransformsRoundTrip:
    """Flips y rotaciones de 90 grados son permutaciones exactas de la
    grilla -- el heatmap recuperado debe ser BIT-IDENTICO al original, no
    solo "cerca". Comparar contra `decode_heatmap` del propio heatmap sin
    transformar (no contra el centro gaussiano nominal): `soft_argmax_2d` es
    un promedio ponderado GLOBAL y tiene un sesgo propio de ~0.05-0.1px en
    centros no enteros incluso sin ningun transform de por medio (confirmado
    corriendo el caso `identity` primero) -- comparar contra el centro
    nominal mezclaria ese sesgo del decoder con el error real del roundtrip
    de la transformacion, que es lo unico que este test debe medir."""

    TOLERANCE_PX = 1e-4

    def _check(self, transform, center_xy):
        canonical = _synthetic_logits(center_xy)
        # Simula "lo que un modelo perfectamente equivariante habria predicho
        # para la vista transformada": aplica la MISMA transformacion
        # geometrica que se le aplicaria a la imagen de entrada.
        transformed_view = transform.transform_image(canonical)
        recovered = transform.untransform_heatmap(transformed_view)
        assert torch.allclose(recovered, canonical, atol=1e-5), (
            f"{transform.name}: recovered heatmap is not bit-identical to the "
            f"untransformed one (max abs diff="
            f"{(recovered - canonical).abs().max().item():.6f})"
        )
        decoded_recovered = _decode_xy(recovered)
        decoded_canonical = _decode_xy(canonical)
        error = torch.linalg.vector_norm(decoded_recovered - decoded_canonical).item()
        assert error < self.TOLERANCE_PX, (
            f"{transform.name}: roundtrip error {error:.6f}px >= {self.TOLERANCE_PX}px "
            f"(decoded_recovered={decoded_recovered.tolist()}, "
            f"decoded_canonical={decoded_canonical.tolist()})"
        )

    def test_identity(self, center_xy):
        transform = default_flip_rotation_transforms()[0]
        assert transform.name == "identity"
        self._check(transform, center_xy)

    def test_hflip(self, center_xy):
        transform = default_flip_rotation_transforms()[1]
        assert transform.name == "hflip"
        self._check(transform, center_xy)

    def test_vflip(self, center_xy):
        transform = default_flip_rotation_transforms()[2]
        assert transform.name == "vflip"
        self._check(transform, center_xy)

    @pytest.mark.parametrize("index", [3, 4, 5])
    def test_rotations(self, center_xy, index):
        transform = default_flip_rotation_transforms()[index]
        assert transform.name in ("rot90", "rot180", "rot270")
        self._check(transform, center_xy)


def test_hflip_actually_moves_the_peak():
    """Guarda contra un bug donde `hflip` fuera accidentalmente una identidad
    (p.ej. flipear el eje equivocado, que en un heatmap cuadrado con centro
    no simetrico produciria un heatmap DISTINTO al original -- si esto
    fallara en silencio con `dim=-2` en vez de `dim=-1` para "horizontal",
    este test seguiria pasando el roundtrip pero estaria flipeando vertical,
    no horizontal). Verifica explicitamente el eje."""
    transform = default_flip_rotation_transforms()[1]
    canonical = _synthetic_logits((10.0, 40.0))
    flipped = transform.transform_image(canonical)
    decoded_flipped = _decode_xy(flipped)
    # hflip cambia x -> (W-1-x), deja y intacto.
    assert abs(decoded_flipped[0].item() - (HEATMAP_SIZE - 1 - 10.0)) < 1e-3
    assert abs(decoded_flipped[1].item() - 40.0) < 1e-3


def test_vflip_actually_moves_the_peak():
    transform = default_flip_rotation_transforms()[2]
    canonical = _synthetic_logits((10.0, 40.0))
    flipped = transform.transform_image(canonical)
    decoded_flipped = _decode_xy(flipped)
    assert abs(decoded_flipped[0].item() - 10.0) < 1e-3
    assert abs(decoded_flipped[1].item() - (HEATMAP_SIZE - 1 - 40.0)) < 1e-3


def test_rotations_are_distinct_from_each_other_and_from_identity():
    """Otra guarda contra transformaciones que colapsen accidentalmente a la
    misma permutacion (p.ej. copiar mal `k` entre rot90/rot180/rot270)."""
    canonical = _synthetic_logits((10.0, 45.0))
    decoded_by_name = {}
    for transform in default_flip_rotation_transforms():
        view = transform.transform_image(canonical)
        decoded_by_name[transform.name] = _decode_xy(view)
    coords = list(decoded_by_name.values())
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            assert torch.linalg.vector_norm(coords[i] - coords[j]).item() > 1.0, (
                "two transforms produced the same peak location -- "
                f"{list(decoded_by_name.keys())[i]} vs {list(decoded_by_name.keys())[j]}"
            )


@pytest.mark.parametrize("scale", [0.9, 1.1])
@pytest.mark.parametrize("center_xy", [(32.0, 32.0), (25.0, 40.0)])
def test_zoom_roundtrip_recovers_coordinate(scale, center_xy):
    """El zoom pasa por dos re-muestreos bilineales (ida + vuelta), asi que
    el roundtrip no es exacto -- pero debe quedar sub-pixel para centros no
    demasiado cerca del borde (donde el zoom-in recorta la region)."""
    transform = default_scale_transforms((scale,))[0]
    canonical = _synthetic_logits(center_xy)
    transformed_view = transform.transform_image(canonical)
    recovered = transform.untransform_heatmap(transformed_view)
    decoded_recovered = _decode_xy(recovered)
    decoded_canonical = _decode_xy(canonical)
    error = torch.linalg.vector_norm(decoded_recovered - decoded_canonical).item()
    assert error < 1.5, f"zoom {scale}: roundtrip error {error:.4f}px >= 1.5px"


def test_zoom_transforms_are_named_distinctly():
    names = [t.name for t in default_scale_transforms((0.9, 1.1))]
    assert names == ["zoom_0.90", "zoom_1.10"]
    assert len(set(names)) == len(names)


def test_zoom_rejects_non_positive_scale():
    from fido.tta_keypoint import _zoom_transform
    with pytest.raises(ValueError):
        _zoom_transform(0.0)
    with pytest.raises(ValueError):
        _zoom_transform(-1.0)


# ---------------------------------------------------------------------------
# Integracion con un modelo real (sin entrenar): verifica que el pipeline
# completo (transformar imagen -> forward -> des-transformar heatmap ->
# promediar -> decodificar una vez) no tiene bugs de forma/dispositivo, y que
# con una sola vista "identity" reproduce EXACTAMENTE lo que el propio
# `model.forward()` decodifica (ningun transform extra se cuela).
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = Task1KeypointModel(base_channels=4, n_downsamples=2)
    model.eval()
    return model


def test_identity_only_tta_matches_plain_forward(tiny_model):
    fundus = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        plain = tiny_model(fundus, fundus_size=64)
    identity_transform = default_flip_rotation_transforms()[:1]
    tta_xy, tta_logits = tta_predict_keypoint(tiny_model, fundus, identity_transform,
                                              fundus_size=64)
    assert torch.allclose(tta_xy, plain["keypoint"], atol=1e-4)
    assert torch.allclose(tta_logits, plain["heatmap_logits"], atol=1e-5)


def test_tta_predict_keypoint_output_shape(tiny_model):
    fundus = torch.rand(3, 3, 64, 64)
    transforms = default_flip_rotation_transforms()
    xy, logits = tta_predict_keypoint(tiny_model, fundus, transforms, fundus_size=64)
    assert xy.shape == (3, 2)
    assert logits.shape[0] == 3 and logits.shape[1] == 1
    assert torch.isfinite(xy).all()
    assert torch.isfinite(logits).all()


def test_tta_forward_logits_rejects_empty_transform_list(tiny_model):
    fundus = torch.rand(1, 3, 64, 64)
    with pytest.raises(ValueError):
        tta_forward_logits(tiny_model, fundus, [], fundus_size=64)


def test_combined_flip_rotation_and_scale_transforms_run_end_to_end(tiny_model):
    fundus = torch.rand(1, 3, 64, 64)
    transforms = default_flip_rotation_transforms() + default_scale_transforms((0.9, 1.1))
    assert len(transforms) == 8
    xy, logits = tta_predict_keypoint(tiny_model, fundus, transforms, fundus_size=64)
    assert xy.shape == (1, 2)
    assert torch.isfinite(xy).all()
    assert torch.isfinite(logits).all()


def test_ensemble_of_identical_models_matches_single_model(tiny_model):
    """Promediar un modelo consigo mismo (dos referencias al mismo checkpoint)
    debe dar exactamente lo que da un solo modelo -- si esto fallara habria
    un bug de doble-conteo/normalizacion en `ensemble_forward_logits`."""
    fundus = torch.rand(2, 3, 64, 64)
    transforms = default_flip_rotation_transforms()
    single_xy, single_logits = tta_predict_keypoint(tiny_model, fundus, transforms,
                                                     fundus_size=64)
    ensemble_xy, ensemble_logits = ensemble_predict_keypoint(
        [tiny_model, tiny_model], fundus, transforms, fundus_size=64,
        temperature=float(tiny_model.heatmap_temperature),
    )
    assert torch.allclose(ensemble_xy, single_xy, atol=1e-4)
    assert torch.allclose(ensemble_logits, single_logits, atol=1e-5)


def test_ensemble_predict_keypoint_rejects_empty_model_list():
    with pytest.raises(ValueError):
        ensemble_predict_keypoint([], torch.rand(1, 3, 64, 64), default_flip_rotation_transforms())
