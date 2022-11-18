from typing import NamedTuple

from elfragmentador.data import datamodules


def check_type(i: NamedTuple):
    expect_names = {
        "seq",
        "mods",
        "spectra",
        "charge",
        "irt",
        "nce",
        "weight",
    }
    assert expect_names == set(i._fields)


def test_dataset_outputs_correct_type(shared_datadir):
    dm = datamodules.TrainingDataModule(10, base_dir=shared_datadir / "parquet")
    check_type(next(iter(dm.train_dataloader())))
    check_type(next(iter(dm.test_dataloader())))
    check_type(next(iter(dm.val_dataloader())))
