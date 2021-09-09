import warnings
import pytest
from elfragmentador.model import PepTransformerModel
from elfragmentador import datamodules
from elfragmentador.utils import prepare_fake_tensor_dataset

from pytorch_lightning import Trainer
import torch


@pytest.fixture(params=["csv", "csv.gz"])
def datamodule(shared_datadir, request):
    path = {
        "csv": shared_datadir / "train_data_sample",
        "csv.gz": shared_datadir / "train_data_sample_compressed",
    }
    datamodule = datamodules.PeptideDataModule(
        batch_size=2, base_dir=path[request.param]
    )
    datamodule.setup()
    return datamodule


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


@pytest.fixture
def checkpoint(tmp_path_factory, tiny_model, shared_datadir):
    warnings.filterwarnings("ignore", ".*The number of training samples .*")
    warnings.filterwarnings(
        "ignore", ".*peaks were annotated The number of training samples .*"
    )
    warnings.filterwarnings(
        "ignore",
        ".*peaks were annotated The number of training samples Skipping peptide due few peaks being annotated .*",
    )
    datamodule = datamodules.PeptideDataModule(
        batch_size=2, base_dir=shared_datadir / "train_data_sample"
    )
    datamodule.setup()
    out = tmp_path_factory.mktemp("data") / "ckpt.ckpt"
    trainer = Trainer(max_epochs=1)
    trainer.fit(tiny_model, datamodule)

    trainer.save_checkpoint(out)
    return out


@pytest.fixture(scope="session")
def tiny_model_ts(tiny_model_builder):
    tiny_model = tiny_model_builder()
    with torch.no_grad():
        ts = tiny_model.to_torchscript()
        ts.eval()
    return ts


@pytest.fixture(scope="session")
def model_pair_builder(tiny_model_builder):
    def model_pair_builder():
        tiny_model = tiny_model_builder()
        ts = tiny_model.to_torchscript()
        return {"base": tiny_model, "traced": ts}

    return model_pair_builder
