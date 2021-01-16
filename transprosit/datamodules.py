from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pandas.core.indexes import base
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transprosit import constants


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        print("\n>>> Initalizing Dataset")
        self.df = df

        sequence_encodings = [eval(x) for x in self.df["SequenceEncoding"]]
        sequence_encodings = [
            x + ([0] * (constants.MAX_SEQUENCE - len(x))) for x in sequence_encodings
        ]
        self.sequence_encodings = [torch.Tensor(x).long().T for x in sequence_encodings]

        spectra_encodings = [eval(x) for x in self.df["SpectraEncoding"]]
        spectra_encodings = [
            x + ([0] * (constants.NUM_FRAG_EMBEDINGS - len(x)))
            for x in spectra_encodings
        ]
        spectra_encodings = [torch.Tensor(x).float().T for x in spectra_encodings]

        self.spectra_encodings = [
            torch.where(x > 0.0, x, torch.Tensor([-1.0])) for x in spectra_encodings
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # encoded_pept = torch.Tensor(eval(self.df.iloc[index].Encoding)).long().T
        # norm_irt = torch.Tensor([self.df.iloc[index].mIRT / 100]).float()
        encoded_sequence = self.sequence_encodings[index]
        encoded_spectra = self.spectra_encodings[index]
        norm_irt = self.norm_irts[index]
        charge = self.charges[index]
        return encoded_sequence, charge, encoded_spectra, norm_irt


class PeptideDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, base_dir="."):
        super().__init__()
        self.batch_size = batch_size
        base_dir = Path(base_dir)

        train_path = base_dir / "combined_train.csv"
        val_path = base_dir / "combined_val.csv"

        assert train_path.exists(), f"File '{train_path}' not found"
        assert val_path.exists(), f"File '{val_path}' not found"

        train_df = pd.read_csv(str(train_path))
        print(train_df)
        print(list(train_df))
        val_df = pd.read_csv(str(val_path))
        print(val_df)
        print(list(val_df))

        print(
            f"Removing Large sequences, currently Train: {len(train_df)} Val: {len(val_df)}"
        )

        train_df = train_df[
            [
                len(eval(x)) <= constants.MAX_SEQUENCE
                for x in train_df["SequenceEncoding"]
            ]
        ].copy()
        val_df = val_df[
            [len(eval(x)) <= constants.MAX_SEQUENCE for x in val_df["SequenceEncoding"]]
        ].copy()

        print(f"Left Train: {len(train_df)} Val: {len(val_df)}")

        self.train_df = train_df
        self.val_df = val_df

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_dir", type=str, default=".")
        return parser

    def setup(self):
        self.train_dataset = PeptideDataset(self.train_df)
        self.val_dataset = PeptideDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, num_workers=0, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
