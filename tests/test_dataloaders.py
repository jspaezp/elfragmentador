from torch.utils.data import DataLoader
from transprosit import datamodules
from pathlib import Path
import pandas as pd
import numpy as np


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


def test_dataloader_handles_missing(shared_datadir):
    base_dataloader_handles_missing(shared_datadir)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    base_dataloader_handles_missing(str(parent_dir) + "/data/")
