import pytest
import random

from torch.utils.data import dataloader
import elfragmentador

from elfragmentador.utils import get_random_peptide
import tempfile
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from elfragmentador import model
from elfragmentador import datamodules
from elfragmentador import constants


def test_concat_encoder():
    x = torch.zeros((5, 2, 20))
    encoder = model.ConcatenationEncoder(10, 0.1, 10)
    output = encoder(x, torch.tensor([[7], [4]]))
    assert output.shape == torch.Size([5, 2, 30]), output.shape
    output = encoder(torch.zeros((5, 1, 20)), torch.tensor([[7]]))
    assert output.shape == torch.Size([5, 1, 30]), output.shape
    print(output)


def test_concat_encoder_adds_right_number():
    x = torch.zeros((5, 1, 20))
    for d in range(1, 10, 1):
        encoder = model.ConcatenationEncoder(d, 0, 200)
        out = encoder(x, torch.tensor([[7]]), debug=True)
        dim_diff = out.shape[-1] - 20
        assert (
            d == dim_diff
        ), "Concatenation Encoder does not add the right number of dims"


def mod_forward_base(datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    print(mod)
    datamodule = datamodules.PeptideDataModule(
        batch_size=5, base_dir=datadir / "train_data_sample"
    )
    datamodule.setup()

    for x in datamodule.val_dataloader():
        print(x)
        break

    print(f">> Shape of inputs {[y.shape for y in x]}")

    with torch.no_grad():
        yhat_irt, yhat_spectra = mod.forward(
            src=x.encoded_sequence,
            charge=x.charge,
            mods=x.encoded_mods,
            nce=x.nce,
            debug=True,
        )

    assert not all(torch.isnan(yhat_irt)), print(yhat_irt.mean())
    assert not all(torch.isnan(yhat_spectra).flatten()), print(yhat_spectra.mean())

    print(f">> Shape of outputs {yhat_irt.shape}, {yhat_spectra.shape}")


def test_model_forward_seq():
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    with torch.no_grad():
        out = mod.predict_from_seq("AAAACDMK", 3, nce=27.0, debug=True)
    print(f">> Shape of outputs {[y.shape for y in out]}")


def _test_export_onnx(datadir, keep=False):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    datamodule = datamodules.PeptideDataModule(
        batch_size=5, base_dir=datadir / "train_data_sample"
    )
    datamodule.setup()

    for input_sample in datamodule.val_dataloader():
        break

    with tempfile.TemporaryDirectory() as tmpdirname:
        if keep:
            tmpdirname = Path("")
        else:
            tmpdirname = Path(tmpdirname)

        print("Exporting to onnx")
        # https://github.com/pytorch/pytorch/issues/22488#issuecomment-630140460
        dummy_input = {"src": input_sample[0], "charge": input_sample[1]}
        print(dummy_input)

        input_names = ["src", "charge"]
        output_names = ["irt", "spectrum"]

        mod.to_onnx(
            tmpdirname / "mod.onnx",
            dummy_input,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
        )


def base_export_torchscript(datadir, keep=False):
    # TODO make this a class method ...
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    mod.decoder.nce_encoder.static_size = constants.NUM_FRAG_EMBEDINGS
    mod.decoder.charge_encoder.static_size = constants.NUM_FRAG_EMBEDINGS

    datamodule = datamodules.PeptideDataModule(
        batch_size=5, base_dir=datadir / "train_data_sample"
    )
    datamodule.setup()

    for input_sample in datamodule.val_dataloader():
        break

    dummy_input = tuple(
        [
            input_sample.encoded_sequence,
            input_sample.nce,
            input_sample.encoded_mods,
            input_sample.charge,
        ]
    )

    print("Exporting to TorchScript")
    script = mod.to_torchscript()
    with torch.no_grad():
        script_out = script(*dummy_input)
        out = mod(*dummy_input)

    print(f">> Shape of base output {[y.shape for y in out]}")
    print(f">> Shape of torchscript output {[y.shape for y in script_out]}")
    print(f">> Head of base out \n{[y.flatten()[:5] for y in out]}")
    print(f">> Head of torchscript out \n{[y.flatten()[:5] for y in script_out]}")


def get_ts_model_pair():
    base_mod = model.PepTransformerModel.load_from_checkpoint(
        elfragmentador.DEFAULT_CHECKPOINT
    )
    base_mod.eval()

    script_mod = base_mod.to_torchscript()

    return base_mod, script_mod


model_pairs = get_ts_model_pair()
model_pairs = [
    pytest.param(model_pairs[0], id="base"),
    pytest.param(model_pairs[1], id="traced"),
]


def concat_batches(batches):
    out = []
    for i, _ in enumerate(batches[0]):
        out.append(torch.cat([b[i] for b in batches]))

    return tuple(out)


def prepare_input_tensors(num=50):
    peps = [
        {
            "nce": 20 + (10 * random.random()),
            "charge": random.randint(1, 5),
            "seq": get_random_peptide(),
        }
        for _ in range(num)
    ]

    tensors = [model.PepTransformerModel.torch_batch_from_seq(**pep) for pep in peps]
    tensors = TensorDataset(*concat_batches(batches=tensors))

    return tensors


input_tensors = prepare_input_tensors(50)


def test_ts_and_base_give_same_result():
    # TODO make parametrized

    base_mod, script_mod = get_ts_model_pair()
    batches = DataLoader(input_tensors, batch_size=1)

    with torch.no_grad():
        for script_batch in batches:
            base_out = base_mod(*script_batch)
            script_out = script_mod(*script_batch)

            for a, b in zip(base_out, script_out):
                assert torch.all(a == b)


@pytest.mark.parametrize("model", model_pairs)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 10, 50])
@pytest.mark.benchmark(min_rounds=10, disable_gc=True, warmup=False)
def test_benchmark_inference_speeds(model, batch_size, benchmark):
    def inf_model():
        batches = DataLoader(input_tensors, batch_size=batch_size)
        for b in batches:
            model(*b)

    with torch.no_grad():
        benchmark(inf_model)


def model_exports_base(datadir, keep=False):
    # test_export_onnx(datadir, keep)
    base_export_torchscript(datadir, keep)


# Disabled for now
def test_model_export(shared_datadir):
    model_exports_base(shared_datadir, keep=False)


def test_model_forward(shared_datadir):
    mod_forward_base(shared_datadir)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    mod_forward_base(str(parent_dir) + "/data/")
    test_model_forward_seq()
    # unable to export right now ...
    base_export_torchscript(str(parent_dir) + "/data/")
