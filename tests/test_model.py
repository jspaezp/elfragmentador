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

    out = mod(x[0], x[1], debug=True)
    print(f">> Shape of outputs {[y.shape for y in out]}")


def test_model_forward(shared_datadir):
    mod_forward_base(shared_datadir)


if __name__ == "__main__":
    mod_forward_base("./tests/data/")
