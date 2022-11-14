import random

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data.dataset import TensorDataset

from elfragmentador.data import datamodules
from elfragmentador.model import PepTransformerModel


def prepare_fake_tensor_dataset(num=50):
    peps = [
        {
            "nce": 20 + (10 * random.random()),
            "charge": random.randint(1, 5),
            "seq": get_random_peptide(),
        }
        for _ in range(num)
    ]

    tensors = [torch_batch_from_seq(**pep) for pep in peps]
    tensors = TensorDataset(*_concat_batches(batches=tensors))

    return tensors


@pytest.fixture(scope="session")
def fake_tensors():
    input_tensors = prepare_fake_tensor_dataset(50)
    return input_tensors


@pytest.fixture(scope="session")
def tiny_model_builder():
    def tiny_model_builder():
        mod = PepTransformerModel(
            num_decoder_layers=3,
            num_encoder_layers=2,
            nhid=112,
            d_model=112,
            nhead=2,
            dropout=0,
            lr=1e-4,
            scheduler="cosine",
            loss_ratio=1000,
            lr_ratio=10,
        )
        mod.eval()
        return mod

    return tiny_model_builder


@pytest.fixture(scope="session")
def tiny_model(tiny_model_builder):
    return tiny_model_builder()
