import pytest
import torch

import elfragmentador.metrics as EFM


@pytest.fixture
def batch(datamodule):
    batch = next(datamodule.train_dataloader().__iter__())
    return batch


def test_spectral_angle_loss(batch):
    loss = EFM.SpectralAngleLoss(dim=1, eps=1e-3)

    out = loss(batch.spectra, batch.spectra)
    assert torch.all(out < 0.001)

    out = loss(batch.spectra, batch.spectra + 10 * torch.rand_like(batch.spectra))
    assert torch.all(out > 0.001)


def test_cosine_loss(batch):
    loss = EFM.CosineLoss(dim=1, eps=1e-4)

    out = loss(batch.spectra, batch.spectra)
    assert torch.all(out < 0.0001)
    out = loss(batch.spectra, batch.spectra + torch.rand_like(batch.spectra))
    assert torch.all(out > 0.0001)


def test_metrics_give_same_result(batch):
    # They are not meant to ...
    loss1 = EFM.SpectralAngleLoss(dim=1, eps=1e-4)
    loss2 = EFM.CosineLoss(dim=1, eps=1e-4)

    spec_base = batch.spectra
    random_spec = torch.rand_like(spec_base)
    spec1 = batch.spectra + random_spec
    spec2 = batch.spectra + 5 * random_spec

    similar_spec_loss = loss1(spec_base, spec1)
    dissimilar_spec_loss = loss1(spec_base, spec2)
    assert torch.all(similar_spec_loss < dissimilar_spec_loss)

    similar_spec_loss = loss2(spec_base, spec1)
    dissimilar_spec_loss = loss2(spec_base, spec2)
    assert torch.all(similar_spec_loss < dissimilar_spec_loss)
