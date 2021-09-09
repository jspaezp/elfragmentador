import pytest
import random

import elfragmentador

import tempfile
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader

from elfragmentador import model
from elfragmentador import datamodules
from elfragmentador import constants
from elfragmentador.model import ForwardBatch

from elfragmentador import utils as efu
from elfragmentador import encoding_decoding as efe


def test_concat_encoder():
    x = torch.zeros((5, 2, 20))
    encoder = model.ConcatenationEncoder(dims_add=10, max_val=10)
    output = encoder(x, torch.tensor([[7], [4]]))
    assert output.shape == torch.Size([5, 2, 30]), output.shape
    output = encoder(torch.zeros((5, 1, 20)), torch.tensor([[7]]))
    assert output.shape == torch.Size([5, 1, 30]), output.shape
    print(output)


def test_concat_encoder_adds_right_number():
    x = torch.zeros((5, 1, 20))
    for d in range(1, 10, 1):
        encoder = model.ConcatenationEncoder(d, 200)
        out = encoder(x, torch.tensor([[7]]), debug=True)
        dim_diff = out.shape[-1] - 20
        assert (
            d == dim_diff
        ), "Concatenation Encoder does not add the right number of dims"


def mod_forward_base(datadir):
    mod = model.PepTransformerModel(nhead=4, d_model=64)
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
            seq=x.seq,
            charge=x.charge,
            mods=x.mods,
            nce=x.nce,
            debug=True,
        )

    assert not all(torch.isnan(yhat_irt)), print(yhat_irt.mean())
    assert not all(torch.isnan(yhat_spectra).flatten()), print(yhat_spectra.mean())

    print(f">> Shape of outputs {yhat_irt.shape}, {yhat_spectra.shape}")


def test_model_forward_seq():
    mod = model.PepTransformerModel(nhead=4, d_model=64)
    with torch.no_grad():
        out = mod.predict_from_seq("AAAACDMK", 3, nce=27.0, debug=True)
    print(f">> Shape of outputs {[y.shape for y in out]}")


def _test_export_onnx(datadir, keep=False):
    mod = model.PepTransformerModel(nhead=4, d_model=64)
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
        dummy_input = input_sample._asdict()
        print(dummy_input)

        input_names = ["seq", "mods", "charge", "nce"]
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
    mod = model.PepTransformerModel(nhead=4, d_model=64)
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
            input_sample.seq,
            input_sample.mods,
            input_sample.charge,
            input_sample.nce,
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


def test_ts_and_base_give_same_result(fake_tensors, model_pair_builder):
    # TODO make parametrized

    model_pair = model_pair_builder()
    base_mod, script_mod = model_pair["base"], model_pair["traced"]
    batches = DataLoader(fake_tensors, batch_size=1)

    with torch.no_grad():
        for script_batch in batches:
            base_out = base_mod(*script_batch)
            script_out = script_mod(*script_batch)

            assert torch.all(script_out[0] == base_out[0])
            assert torch.all(script_out[1] == base_out[1])


@pytest.mark.parametrize("model", ["base", "traced"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 10])
@pytest.mark.benchmark(min_rounds=10, disable_gc=True, warmup=False)
def test_benchmark_inference_speeds(
    model, batch_size, fake_tensors, model_pair, benchmark
):
    model = model_pair[model]

    def inf_model():
        batches = DataLoader(fake_tensors, batch_size=batch_size)
        for b in batches:
            model(*b)

    with torch.no_grad():
        benchmark(inf_model)


def model_exports_base(datadir, keep=False):
    # Disabled for now
    # test_export_onnx(datadir, keep)
    base_export_torchscript(datadir, keep)


def test_model_export(shared_datadir):
    model_exports_base(shared_datadir, keep=False)


def test_model_forward(shared_datadir):
    mod_forward_base(shared_datadir)


@pytest.fixture
def setup_model(tiny_model):
    mod = tiny_model.eval()

    # TODO make this a test ...
    seqs = [efu.get_random_peptide() for _ in range(100)]
    charges = [torch.tensor([[random.randint(2, 6)]]).long() for _ in seqs]
    nces = [torch.tensor([[random.uniform(20, 40)]]).float() for _ in seqs]

    return tiny_model, seqs, charges, nces


def test_variable_length_has_same_results(setup_model):
    mod, seqs, charges, nces = setup_model
    with torch.no_grad():
        for s, c, n in zip(seqs, charges, nces):
            batch1 = efe.encode_mod_seq(seq=s, pad_zeros=True)
            batch2 = efe.encode_mod_seq(seq=s, pad_zeros=False)

            batch1 = ForwardBatch(
                seq=torch.unsqueeze(torch.tensor(batch1.aas), 0),
                nce=n,
                mods=torch.unsqueeze(torch.tensor(batch1.mods), 0),
                charge=c,
            )

            batch2 = ForwardBatch(
                seq=torch.unsqueeze(torch.tensor(batch2.aas), 0),
                nce=n,
                mods=torch.unsqueeze(torch.tensor(batch2.mods), 0),
                charge=c,
            )

            assert torch.all(
                (
                    mod.forward(**batch1._asdict(), debug=True).spectra
                    - mod.forward(**batch2._asdict(), debug=True).spectra
                )
                < 1e-5
            )


@pytest.mark.parametrize("variable_length", [True, False])
def test_variable_length(benchmark, variable_length, setup_model):
    # No zero padding means variable length ...
    mod, seqs, charges, nces = setup_model

    @torch.no_grad()
    def setup_batches(seqs, charges, nces, pad_zeros):
        batches = []
        for s, c, n in zip(seqs, charges, nces):
            batch = efe.encode_mod_seq(seq=s, pad_zeros=pad_zeros)
            batch = ForwardBatch(
                seq=torch.unsqueeze(torch.tensor(batch.aas), 0),
                nce=n,
                mods=torch.unsqueeze(torch.tensor(batch.mods), 0),
                charge=c,
            )
            batches.append(batch)

        return batches

    @torch.no_grad()
    def main():
        for batch in batches:
            mod.forward(*batch)

    batches = setup_batches(
        seqs=seqs, charges=charges, nces=nces, pad_zeros=not variable_length
    )
    benchmark(main)
