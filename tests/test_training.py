import math
from pathlib import Path

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from elfragmentador import datamodules, model


def get_datamodule(datadir):
    datamodule = datamodules.PeptideDataModule(batch_size=5, base_dir=datadir)
    datamodule.setup()
    return datamodule


def mod_train_base(datadir):
    datamodule = get_datamodule(datadir)
    mod = model.PepTransformerModel(nhead=4, ninp=64)

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(mod, datamodule)

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(mod, datamodule)


def base_train_works_on_schdulers(datadir):
    datamodule = get_datamodule(datadir)

    for sch in model.PepTransformerModel.accepted_schedulers[::-1]:
        print(f"\n\n\n>>>>>>>>>>>>>>>>> {sch} \n\n\n")
        lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor(
            logging_interval="step"
        )
        trainer = pl.Trainer(
            max_epochs=10, callbacks=[lr_monitor], limit_train_batches=1
        )
        mod = model.PepTransformerModel(nhead=4, ninp=64, scheduler=sch)
        mod.steps_per_epoch = math.ceil(
            len(datamodule.train_dataset) / datamodule.batch_size
        )
        trainer.fit(mod, datamodule)


def test_train_works_on_schedulers(shared_datadir):
    base_train_works_on_schdulers(shared_datadir / "train_data_sample")


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
    mod_train_base(shared_datadir / "train_data_sample")


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    datadir = str(parent_dir) + "/data/"
    mod_train_base(datadir)
    mod_train_with_missing(datadir)
    base_train_works_on_schdulers(datadir)
