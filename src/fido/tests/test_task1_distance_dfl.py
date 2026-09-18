import numpy as np
import pytest
import torch

from fido.models.task1_distance_dfl import (
    BscanSliceEncoder,
    DistanceDFLHead,
    Task1DistanceDFLModel,
    dfl_loss,
    load_pretrained_segmentation_encoder,
)

D_MIN, D_MAX, REG_MAX = -20.0, 320.0, 64


def _head() -> DistanceDFLHead:
    return DistanceDFLHead(in_features=8, reg_max=REG_MAX, d_min=D_MIN, d_max=D_MAX)


# --- round-trip: encode -> decode recupera el target (identidad exacta) ---

@pytest.mark.parametrize("target_value", [-20.0, -5.5, 0.0, 12.3, 100.0, 169.4, 319.999, 320.0])
def test_encode_decode_round_trip_recovers_target(target_value):
    head = _head()
    target = torch.tensor([target_value], dtype=torch.float32)
    distribution = head.target_distribution(target)
    # `target_distribution` es "dos-calientes": su esperanza ponderada, por
    # construccion algebraica, es EXACTAMENTE `encode_target(target)` -- el
    # round-trip debe ser ~0, no solo "cercano por casualidad".
    logits = torch.log(distribution.clamp(min=1e-12))
    decoded = head.decode(logits)
    assert torch.abs(decoded - target).item() < 1e-3


def test_round_trip_error_stays_near_zero_across_random_targets():
    head = _head()
    generator = torch.Generator().manual_seed(0)
    targets = torch.empty(200).uniform_(D_MIN, D_MAX, generator=generator)
    distribution = head.target_distribution(targets)
    logits = torch.log(distribution.clamp(min=1e-12))
    decoded = head.decode(logits)
    errors = (decoded - targets).abs()
    assert errors.mean().item() < 1e-3
    assert errors.max().item() < 1e-2


# --- la loss baja en un caso sintetico ---

def test_dfl_loss_decreases_when_optimizing_a_fixed_target():
    torch.manual_seed(0)
    head = _head()
    features = torch.randn(4, 8)
    target = torch.full((4,), 150.0)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)

    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        logits = head(features)
        loss = dfl_loss(logits, target, head)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]
    with torch.no_grad():
        decoded = head.decode(head(features))
    assert torch.abs(decoded - target).mean().item() < 15.0  # ~1 bin de ancho


# --- targets fuera de rango no rompen nada ---

@pytest.mark.parametrize("out_of_range_value", [-1e4, -100.0, 1e4, 5000.0])
def test_out_of_range_target_does_not_break_loss_or_decode(out_of_range_value):
    head = _head()
    features = torch.randn(3, 8)
    target = torch.tensor([out_of_range_value, 50.0, D_MAX + 1.0])
    logits = head(features)
    loss = dfl_loss(logits, target, head)
    assert torch.isfinite(loss)
    decoded = head.decode(logits)
    assert torch.isfinite(decoded).all()
    # El target extremo se satura al bin borde, no explota fuera de rango.
    encoded = head.encode_target(target)
    assert (encoded >= 0.0).all() and (encoded <= REG_MAX).all()


def test_encode_target_clamps_extremes_to_bin_boundaries():
    head = _head()
    below = head.encode_target(torch.tensor([D_MIN - 1000.0]))
    above = head.encode_target(torch.tensor([D_MAX + 1000.0]))
    assert below.item() == pytest.approx(0.0)
    assert above.item() == pytest.approx(float(REG_MAX))


# --- constructor valida hiperparametros ---

def test_head_rejects_invalid_range_and_reg_max():
    with pytest.raises(ValueError):
        DistanceDFLHead(in_features=4, reg_max=0, d_min=0.0, d_max=1.0)
    with pytest.raises(ValueError):
        DistanceDFLHead(in_features=4, reg_max=8, d_min=5.0, d_max=5.0)


# --- modelo completo: forma de salida y ausencia de NaN, incluso con
#     entrada en cero (caso sin OCT, `Task1Dataset` rellena con ceros) ---

def test_full_model_forward_shapes_and_handles_zero_input():
    torch.manual_seed(0)
    model = Task1DistanceDFLModel(reg_max=32, d_min=D_MIN, d_max=D_MAX,
                                   base_channels=4, depth=2)
    model.eval()
    bscan = torch.zeros(2, 2, 32, 32)  # (B=2, slices=2, H=32, W=32)
    with torch.no_grad():
        out = model(bscan)
    assert out["logits"].shape == (2, 33)
    assert out["distance"].shape == (2,)
    assert torch.isfinite(out["distance"]).all()


def test_full_model_rejects_wrong_slice_count():
    model = Task1DistanceDFLModel(reg_max=8, d_min=0.0, d_max=10.0,
                                   base_channels=4, depth=1)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 16, 16))


# --- transferencia de pesos del encoder de segmentacion ya entrenado ---

def test_load_pretrained_segmentation_encoder_copies_matching_shapes():
    encoder = BscanSliceEncoder(in_channels=1, base_channels=8, depth=2)
    # Fingimos un state_dict de `unet_bscan_seg.UNet` con las mismas claves
    # de encoder (mismo base_channels/depth) mas claves extra de
    # decoder/bottleneck que este encoder NO tiene -- deben ignorarse sin
    # romper la carga.
    fake_state = {k: v.clone() + 1.0 for k, v in encoder.state_dict().items()}
    fake_state["bottleneck.conv1.weight"] = torch.zeros(1)
    fake_state["decoders.0.conv1.weight"] = torch.zeros(1)
    fake_state["final_conv.weight"] = torch.zeros(1)

    n_copied = load_pretrained_segmentation_encoder(encoder, fake_state)

    assert n_copied == len(encoder.state_dict())
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, fake_state[key])


def test_load_pretrained_segmentation_encoder_skips_shape_mismatch():
    encoder = BscanSliceEncoder(in_channels=1, base_channels=8, depth=2)
    mismatched_state = {"encoders.0.conv1.weight": torch.zeros(1, 1, 1, 1)}
    n_copied = load_pretrained_segmentation_encoder(encoder, mismatched_state)
    assert n_copied == 0
