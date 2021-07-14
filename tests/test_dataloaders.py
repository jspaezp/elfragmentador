from typing import NamedTuple
import torch
from torch.utils.data import DataLoader
from elfragmentador import constants, datamodules
from pathlib import Path
import pandas as pd
import numpy as np


def check_lengths(i: NamedTuple):
    expect_names = set(
        [
            "encoded_sequence",
            "encoded_mods",
            "encoded_spectra",
            "charge",
            "norm_irt",
            "nce",
        ]
    )
    assert expect_names == set(i._fields)
    assert i.encoded_sequence.shape == torch.Size([constants.MAX_TENSOR_SEQUENCE])
    assert i.encoded_mods.shape == torch.Size([constants.MAX_TENSOR_SEQUENCE])
    assert i.encoded_spectra.shape == torch.Size([constants.NUM_FRAG_EMBEDINGS])
    assert i.charge.shape == torch.Size([1])
    assert i.norm_irt.shape == torch.Size([1])
    assert i.nce.shape == torch.Size([1])


def test_dataset_outputs_correct_dims(shared_datadir):
    df = pd.read_csv(str(shared_datadir) + "/combined_val.csv")
    df = datamodules.filter_df_on_sequences(df)
    dataset = datamodules.PeptideDataset(df)
    i = dataset[0]
    for x in i:
        print(">>>")
        print(x)
        print(type(x))
        print(x.data)

    check_lengths(i)


def test_dataset_from_sptxt_works(shared_datadir):
    infiles = [
        "/small_phospho_spectrast.sptxt",
        "/small_proteome_spectrast.sptxt",
        "/sample.sptxt",
    ]

    for f in infiles:
        ds = datamodules.PeptideDataset.from_sptxt(str(shared_datadir) + f)
        check_lengths(ds[0])


def base_dataloader_handles_missing(datadir):
    df = pd.read_csv(str(datadir) + "/combined_val.csv")
    df = datamodules.filter_df_on_sequences(df)
    df.loc[1, "mIRT"] = np.nan

    print(df)
    ds = datamodules.PeptideDataset(df)
    print(ds[1].norm_irt)
    dl = DataLoader(ds, 4)
    for i in dl:
        print(i.norm_irt)
        print(i.norm_irt[~i.norm_irt.isnan()])

    print(i)


def test_dataloader_handles_missing(shared_datadir):
    base_dataloader_handles_missing(shared_datadir)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    base_dataloader_handles_missing(str(parent_dir) + "/data/")
