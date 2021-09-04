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
            "seq",
            "mods",
            "spectra",
            "charge",
            "irt",
            "nce",
            "weight",
        ]
    )
    assert expect_names == set(i._fields)
    assert i.seq.shape == torch.Size([constants.MAX_TENSOR_SEQUENCE])
    assert i.mods.shape == torch.Size([constants.MAX_TENSOR_SEQUENCE])
    assert i.spectra.shape == torch.Size([constants.NUM_FRAG_EMBEDINGS])
    assert i.charge.shape == torch.Size([1])
    assert i.irt.shape == torch.Size([1])
    assert i.nce.shape == torch.Size([1])
    assert i.weight.shape == torch.Size([1])


def test_dataset_outputs_correct_dims(shared_datadir):
    df = pd.read_csv(str(shared_datadir) + "/train_data_sample/combined_val.csv")
    dataset = datamodules.PeptideDataset(df, filter_df=True)
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
