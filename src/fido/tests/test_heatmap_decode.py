import pytest
import torch

from fido.heatmap_decode import (
    decode_heatmap,
    gaussian_heatmap_target,
    heatmap_to_image_xy,
    image_to_heatmap_xy,
)


def test_decoders_recover_fractional_gaussian():
    center = torch.tensor([[[10.25, 12.50]]])
    probability = gaussian_heatmap_target(center, 32, 32, sigma=1.5)
    logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
    for mode, tolerance in (("local", 0.08), ("dark", 0.08)):
        decoded = decode_heatmap(logits, mode=mode, window=7)
        assert torch.linalg.vector_norm(decoded - center) < tolerance


def test_local_decoder_ignores_remote_mode():
    main = gaussian_heatmap_target(torch.tensor([[[10.25, 12.50]]]), 64, 64, 1.5)
    remote = 0.8 * gaussian_heatmap_target(torch.tensor([[[50.0, 50.0]]]), 64, 64, 1.5)
    logits = torch.logit((main + remote).clamp(1e-6, 1 - 1e-6))
    decoded = decode_heatmap(logits, mode="local", window=7)
    assert torch.linalg.vector_norm(decoded[0, 0] - torch.tensor([10.25, 12.50])) < 0.25


@pytest.mark.parametrize("mode", ["global", "local", "dark"])
def test_decoder_is_finite_for_border_flat_and_nan(mode):
    heatmap = torch.zeros(3, 1, 9, 9)
    heatmap[0, 0, 0, 0] = 10
    heatmap[2, 0, 4, 4] = float("nan")
    decoded = decode_heatmap(heatmap, mode=mode)
    assert torch.isfinite(decoded).all()
    assert (decoded[..., 0] >= 0).all() and (decoded[..., 0] <= 8).all()
    assert (decoded[..., 1] >= 0).all() and (decoded[..., 1] <= 8).all()


def test_udp_round_trip():
    xy = torch.tensor([[0.0, 0.0], [1023.0, 1023.0], [221.3, 617.8]])
    heatmap_xy = image_to_heatmap_xy(xy, (1024, 1024), (64, 64))
    recovered = heatmap_to_image_xy(heatmap_xy, (1024, 1024), (64, 64))
    assert torch.allclose(recovered, xy, atol=1e-5)


def test_udp_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        image_to_heatmap_xy(torch.zeros(1, 2), 1, 64)


def test_dispatcher_rejects_unknown_mode_and_even_window():
    heatmap = torch.zeros(1, 1, 9, 9)
    with pytest.raises(ValueError, match="mode"):
        decode_heatmap(heatmap, mode="unknown")
    with pytest.raises(ValueError, match="odd"):
        decode_heatmap(heatmap, mode="local", window=4)
