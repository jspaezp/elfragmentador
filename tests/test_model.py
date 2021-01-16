import torch

from transprosit import model
from transprosit import datamodules


def mod_forward_base(datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=32)
    print(mod)
    datamodule = datamodules.PeptideDataModule(batch_size=2, base_dir=datadir)
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


def test_model_forward(shared_datadir):
    mod_forward_base(shared_datadir)


if __name__ == "__main__":
    from pathlib import Path

    parent_dir = Path(__file__).parent
    mod_forward_base(str(parent_dir) + "/data/")
    test_model_forward_seq()
