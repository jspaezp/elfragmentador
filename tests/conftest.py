import pytest
from elfragmentador.model import PepTransformerModel
from elfragmentador.utils import prepare_fake_tensor_dataset


@pytest.fixture(scope="session")
def fake_tensors():
    input_tensors = prepare_fake_tensor_dataset(50)
    return input_tensors


@pytest.fixture(scope="session")
def tiny_model():
    mod = PepTransformerModel(
        num_decoder_layers=3, num_encoder_layers=2, nhid=64, ninp=64, nhead=2
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
