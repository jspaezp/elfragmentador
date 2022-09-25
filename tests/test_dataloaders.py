from typing import NamedTuple

import pandas as pd

from elfragmentador import datamodules


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
    df = pd.read_csv(str(shared_datadir) + "/train_data_sample/combined_val.csv")
    dataset = datamodules.PeptideDataset(df, filter_df=True)
    i = dataset[0]
    for x in i:
        print(">>>")
        print(x)
        print(type(x))
        print(x.data)

    check_type(i)


def test_dataset_from_sptxt_works(shared_datadir):
    infiles = [
        "/small_phospho_spectrast.sptxt",
        "/small_proteome_spectrast.sptxt",
        "/sample.sptxt",
    ]

    for f in infiles:
        ds = datamodules.PeptideDataset.from_sptxt(str(shared_datadir) + f)
        check_type(ds[0])
