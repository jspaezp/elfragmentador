from collections import namedtuple
from argparse import ArgumentParser
from pathlib import PosixPath, Path

import pandas as pd
from pandas.core.indexes import base
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transprosit import constants
from pandas.core.frame import DataFrame
from torch.utils.data.dataloader import DataLoader
from typing import Union

train_batch = namedtuple(
    "TrainBatch", "encoded_sequence, charge, encoded_spectra, norm_irt"
)


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, df: DataFrame) -> None:
        super().__init__()
        print("\n>>> Initalizing Dataset")
        self.df = df

        sequence_encodings = [eval(x) for x in self.df["SequenceEncoding"]]
        lengths = [len(x) for x in sequence_encodings]
        unique_lengths = set(lengths)
        match_max = [1 for x in lengths if x == constants.MAX_SEQUENCE]
        print(
            (
                f"{len(match_max)}/{len(sequence_encodings)} "
                f"Sequences actually match the max sequence length of"
                f" {constants.MAX_SEQUENCE},"
                f" found {unique_lengths}"
            )
        )

        sequence_encodings = [
            x + ([0] * (constants.MAX_SEQUENCE - len(x))) for x in sequence_encodings
        ]
        self.sequence_encodings = [torch.Tensor(x).long().T for x in sequence_encodings]

        spectra_encodings = [eval(x) for x in self.df["SpectraEncoding"]]
        lengths = [len(x) for x in spectra_encodings]
        unique_lengths = set(lengths)
        match_max = [1 for x in lengths if x == constants.NUM_FRAG_EMBEDINGS]
        print(
            (
                f"{len(match_max)}/{len(spectra_encodings)} "
                f"Spectra actually match the max spectra length of "
                f"{constants.NUM_FRAG_EMBEDINGS}, "
                f"found {unique_lengths}"
            )
        )

        spectra_encodings = [
            x + ([0] * (constants.NUM_FRAG_EMBEDINGS - len(x)))
            for x in spectra_encodings
        ]
        spectra_encodings = [torch.Tensor(x).float().T for x in spectra_encodings]

        self.spectra_encodings = [
            torch.where(x > 0.01, x, torch.Tensor([0.0])) for x in spectra_encodings
        ]

        spectra_lengths = set([len(x) for x in self.spectra_encodings])
        sequence_lengths = set([len(x) for x in self.sequence_encodings])
        print(
            (
                f"Dataset Initialized with {len(df)} entries."
                f" Spectra length: {spectra_lengths}"
                f" Sequence length: {sequence_lengths}"
            )
        )

        # Pretty sure this last 2 can be optimized vectorizing them
        self.norm_irts = [torch.Tensor([x / 100]).float() for x in self.df["mIRT"]]
        self.charges = [torch.Tensor([x]).long() for x in self.df["Charges"]]

        print(">>> Done Initializing dataset\n")

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
    print(df)
    print(list(df))
    print(f"Removing Large sequences, currently {name}: {len(df)}")
    df = (
        df[[len(eval(x)) <= constants.MAX_SEQUENCE for x in df["SequenceEncoding"]]]
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
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
