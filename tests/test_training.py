import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from pathlib import Path
import pytorch_lightning as pl
from transprosit import datamodules, model


def mod_train_base(datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    print(mod)
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=datadir)
    datamodule.setup()

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(mod, datamodule)

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(mod, datamodule)


def mod_train_with_missing(datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=datadir)
    datamodule.train_df.loc[
        [x for x in range(len(datamodule.train_df))], "mIRT"
    ] = np.nan
    datamodule.val_df.loc[[x for x in range(len(datamodule.val_df))], "mIRT"] = np.nan
    datamodule.setup()

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(mod, datamodule)


def test_model_train(shared_datadir):
    mod_train_base(shared_datadir)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    mod_train_base(str(parent_dir) + "/data/")
    mod_train_with_missing(str(parent_dir) + "/data/")
