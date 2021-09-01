import elfragmentador.metrics as EFM
import pytest
import torch


@pytest.fixture
def batch(datamodule):
    batch = next(datamodule.train_dataloader().__iter__())
    return batch


def test_spectral_angle_loss(batch):
    loss = EFM.SpectralAngleLoss(dim=1, eps=1e-4)
    out = loss(batch.encoded_spectra, batch.encoded_spectra)

    assert torch.all(out < 0.0001)


def test_cosine_loss(batch):
    loss = EFM.CosineLoss(dim=1, eps=1e-4)
    out = loss(batch.encoded_spectra, batch.encoded_spectra)

    assert torch.all(out < 0.0001)


def test_metrics_give_same_result(batch):
    loss1 = EFM.SpectralAngleLoss(dim=1, eps=1e-4)
    loss2 = EFM.CosineLoss(dim=1, eps=1e-4)

    spec1 = batch.encoded_spectra
    spec2 = batch.encoded_spectra + torch.rand_like(spec1)

    out1 = loss1(spec1, spec1 * 2)
    out2 = loss2(spec1, spec1 * 2)

    assert torch.all((out1 - out2) < 0.0001)
