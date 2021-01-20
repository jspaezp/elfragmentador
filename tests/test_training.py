from pathlib import Path
import pytorch_lightning as pl
from transprosit import datamodules, model


def mod_train_base(datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    print(mod)
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=datadir)
    datamodule.setup()

    for x in datamodule.val_dataloader():
        break

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(mod, datamodule)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(mod, datamodule)


def test_model_train(shared_datadir):
    mod_train_base(shared_datadir)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    mod_train_base(str(parent_dir) + "/data/")