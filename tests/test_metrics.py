import pytest
import pytorch_lightning as pl
import torch

import elfragmentador.metrics as EFM


@pytest.fixture
def batch(datamodule):
    batch = next(datamodule.train_dataloader().__iter__())
    return batch


def check_all(x, check="small", context=None):
    try:
        if check == "small":
            assert torch.all(x < 0.001)
        elif check == "large":
            assert torch.all(x > 0.001)
        else:
            raise ValueError(f"Unknown check: {check}")
    except AssertionError:
        raise ValueError(f"{x}\n\n is not all {check}, Context: {context}")


def evaluate_loss(loss, batch, dim=1):
    pl.seed_everything(42)

    out = loss(batch.spectra, batch.spectra)

    expect_size = [x for i, x in enumerate(batch.spectra.shape) if i != dim]
    got_size = list(x for x in out.shape)
    for x, y in zip(expect_size, got_size):
        assert x == y
    check_all(out, check="small", context={"in_spectrum": batch.spectra})

    rand_like_spectrum = torch.rand_like(batch.spectra)
    out = loss(batch.spectra, batch.spectra + 10 * rand_like_spectrum)
    check_all(out, check="large")


def test_spectral_angle_loss(batch):
    loss = EFM.SpectralAngleLoss(dim=1, eps=1e-3)
    evaluate_loss(loss, batch, dim=1)


def test_cosine_loss(batch):
    loss = EFM.CosineLoss(dim=1, eps=1e-4)
    evaluate_loss(loss, batch, dim=1)


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
