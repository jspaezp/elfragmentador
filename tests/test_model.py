import tempfile
from pathlib import Path

import torch

from transprosit import model
from transprosit import datamodules


def mod_forward_base(datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=32)
    print(mod)
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=datadir)
    datamodule.setup()

    for x in datamodule.val_dataloader():
        break

    print(f">> Shape of inputs {[y.shape for y in x]}")

    with torch.no_grad():
        out = mod(x[0], x[1], debug=True)

    print(f">> Shape of outputs {[y.shape for y in out]}")


def test_model_forward_seq():
    mod = model.PepTransformerModel(nhead=4, ninp=32)
    with torch.no_grad():
        out = mod.predict_from_seq("AAAACDMK", 3, debug=True)
    print(out)


def model_exports_base(datadir, keep=False):
    mod = model.PepTransformerModel(nhead=4, ninp=32)
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=datadir)
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

        print("Exporting to TorchScript")
        mod.to_torchscript(
            tmpdirname / "mod.ts", example_inputs=dummy_input, method="trace"
        )


# Disabled for now
def _test_model_export(shared_datadir):
    model_exports_base(shared_datadir, keep=False)


def test_model_forward(shared_datadir):
    mod_forward_base(shared_datadir)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    mod_forward_base(str(parent_dir) + "/data/")
    test_model_forward_seq()
    # unable to export right now ...
    # model_exports_base(str(parent_dir) + "/data/")
