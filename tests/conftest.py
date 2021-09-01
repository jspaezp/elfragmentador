import pytest
from elfragmentador.model import PepTransformerModel
from elfragmentador import datamodules
from elfragmentador.utils import prepare_fake_tensor_dataset


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
def tiny_model():
    mod = PepTransformerModel(
        num_decoder_layers=3,
        num_encoder_layers=2,
        nhid=112,
        ninp=112,
        nhead=2,
        dropout=0,
        lr=1e-4,
        scheduler="cosine",
        loss_ratio=1000,
        lr_ratio=10,
    )
    mod.eval()
    return mod


@pytest.fixture(scope="session")
def tiny_model_ts(tiny_model):
    ts = tiny_model.to_torchscript()
    ts.eval()
    return ts


@pytest.fixture(scope="session")
def model_pair(tiny_model, tiny_model_ts):
    return {"base": tiny_model, "traced": tiny_model_ts}


@pytest.fixture(scope="session")
def tiny_model_chekcpoint(tiny_model, tempdir):
    out_file = tempdir / "out.ckpt"
    tiny_model.save_checkpoint(tempdir, out_file)
    return out_file
