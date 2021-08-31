import pytest

import math
from pathlib import Path

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from elfragmentador import datamodules, model


@pytest.fixture(params=["csv", "csv.gz"])
def datamodule(shared_datadir, request):
    path = {
        "csv": shared_datadir / "train_data_sample", 
        "csv.gz": shared_datadir / "train_data_sample_compressed"
    }
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=path[request.param])
    datamodule.setup()
    return datamodule


def test_mod_train_base(datamodule):
    mod = model.PepTransformerModel(nhead=4, ninp=64)

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(mod, datamodule)

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(mod, datamodule)


@pytest.mark.parametrize("scheduler", model.PepTransformerModel.accepted_schedulers[::-1])
def test_base_train_works_on_schdulers(datamodule, scheduler, tiny_model):
    print(f"\n\n\n>>>>>>>>>>>>>>>>> {scheduler} \n\n\n")
    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor(
        logging_interval="step"
    )
    trainer = pl.Trainer(
        max_epochs=10, callbacks=[lr_monitor], limit_train_batches=1
    )
    mod = tiny_model
    mod.steps_per_epoch = math.ceil(
        len(datamodule.train_dataset) / datamodule.batch_size
    )
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


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    datadir = str(parent_dir) + "/data/"
    mod_train_base(datadir)
    mod_train_with_missing(datadir)
    base_train_works_on_schdulers(datadir)
