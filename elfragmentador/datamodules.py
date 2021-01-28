from collections import namedtuple
from argparse import ArgumentParser
from pathlib import PosixPath, Path

import pandas as pd
from pandas.core.indexes import base
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from elfragmentador import constants, spectra
from pandas.core.frame import DataFrame
from torch.utils.data.dataloader import DataLoader
from typing import Union

train_batch = namedtuple(
    "TrainBatch", "encoded_sequence, charge, encoded_spectra, norm_irt"
)


def match_lengths(nested_list, max_len, name="items", verbose=True):
    lengths = [len(x) for x in nested_list]
    unique_lengths = set(lengths)
    match_max = [1 for x in lengths if x == max_len]

    out_message = (
        f"{len(match_max)}/{len(nested_list)} "
        f"{name} actually match the max sequence length of"
        f" {max_len},"
        f" found {unique_lengths}"
    )
    if verbose:
        print(out_message)

    out = [
        x + ([0] * (max_len - len(x))) if len(x) != max_len else x for x in nested_list
    ]
    out = torch.stack([torch.Tensor(x).T for x in out])
    return out


def match_colnames(df):
    colnames = list(df)
    out = {
        "SeqE": [i for i, x in enumerate(colnames) if "Encoding" in x and "Seq" in x][
            0
        ],
        "SpecE": [i for i, x in enumerate(colnames) if "Encoding" in x and "Spec" in x][
            0
        ],
        "Ch": [i for i, x in enumerate(colnames) if "harg" in x][0],
        "iRT": [i for i, x in enumerate(colnames) if "IRT" in x or "iRT" in x][0],
    }
    out = {k: colnames[v] for k, v in out.items()}
    return out


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, df: DataFrame) -> None:
        super().__init__()
        print("\n>>> Initalizing Dataset")
        self.df = df  # TODO remove this for memmory ...

        name_match = match_colnames(df)

        sequence_encodings = [eval(x) for x in self.df[name_match["SeqE"]]]
        sequence_encodings = match_lengths(
            sequence_encodings, constants.MAX_SEQUENCE, "Sequences"
        )
        self.sequence_encodings = sequence_encodings.long()

        spectra_encodings = [eval(x) for x in self.df[name_match["SpecE"]]]
        spectra_encodings = match_lengths(
            spectra_encodings, constants.NUM_FRAG_EMBEDINGS, "Spectra"
        )
        self.spectra_encodings = spectra_encodings.float()

        spectra_lengths = len(self.spectra_encodings[0])
        sequence_lengths = len(self.sequence_encodings[0])
        print(
            (
                f"Dataset Initialized with {len(df)} entries."
                f" Spectra length: {spectra_lengths}"
                f" Sequence length: {sequence_lengths}"
            )
        )

        # Pretty sure this last 2 can be optimized vectorizing them
        self.norm_irts = (
            torch.Tensor(self.df[name_match["iRT"]] / 100).float().unsqueeze(1)
        )
        self.charges = torch.Tensor(self.df[name_match["Ch"]]).long().unsqueeze(1)

        print(">>> Done Initializing dataset\n")

    @staticmethod
    def from_sptxt(filepath, max_spec=1e6, filter_df=True, *args, **kwargs):
        df = spectra.encode_sptxt(str(filepath), max_spec=max_spec, *args, **kwargs)
        if filter_df:
            df = filter_df_on_sequences(df)

        return PeptideDataset(df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        # encoded_pept = torch.Tensor(eval(self.df.iloc[index].Encoding)).long().T
        # norm_irt = torch.Tensor([self.df.iloc[index].mIRT / 100]).float()
        encoded_sequence = self.sequence_encodings[index]
        encoded_spectra = self.spectra_encodings[index]
        norm_irt = self.norm_irts[index]
        charge = self.charges[index]

        out = train_batch(encoded_sequence, charge, encoded_spectra, norm_irt)
        return out


def filter_df_on_sequences(df, name=""):
    name_match = match_colnames(df)
    print(list(df))
    print(f"Removing Large sequences, currently {name}: {len(df)}")
    df = (
        df[[len(eval(x)) <= constants.MAX_SEQUENCE for x in df[name_match["SeqE"]]]]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Left {name}: {len(df)}")
    return df


class PeptideDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int = 64, base_dir: Union[PosixPath, str] = "."
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        base_dir = Path(base_dir)

        train_path = base_dir / "combined_train.csv"
        val_path = base_dir / "combined_val.csv"

        assert train_path.exists(), f"File '{train_path}' not found"
        assert val_path.exists(), f"File '{val_path}' not found"

        train_df = pd.read_csv(str(train_path))
        train_df = filter_df_on_sequences(train_df)
        val_df = pd.read_csv(str(val_path))
        val_df = filter_df_on_sequences(val_df)

        self.train_df = train_df
        self.val_df = val_df

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_dir", type=str, default=".")
        return parser

    def setup(self) -> None:
        self.train_dataset = PeptideDataset(self.train_df)
        self.val_dataset = PeptideDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, num_workers=0, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
