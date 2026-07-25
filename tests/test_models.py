import torch

from mpd_df.models import EEGNet, MSCNNCAM


def test_deep_models_emit_binary_logits() -> None:
    inputs = torch.randn(3, 8, 200)
    eegnet = EEGNet(n_channels=8, n_times=200)
    mscnn = MSCNNCAM(n_channels=8)
    assert eegnet(inputs).shape == (3, 2)
    assert mscnn(inputs).shape == (3, 2)

