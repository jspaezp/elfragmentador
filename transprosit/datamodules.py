from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pandas.core.indexes import base
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

        self.sequence_encodings = [
            torch.Tensor(eval(x)).long().T for x in self.df["SequenceEncoding"]
        ]
        self.spectra_encodings = [
            torch.Tensor(eval(x)).long().T for x in self.df["SpectraEncoding"]
        ]
        self.spectra_encodings = [
            torch.where(x > 0., x, torch.Tensor([-1.])) for x in self.spectra_encodings
        ]

        # Pretty sure this last 2 can be optimized vectorizing them
        self.norm_irts = [torch.Tensor([x / 100]).float() for x in self.df["mIRT"]]
        self.charges = [torch.Tensor([x]).long() for x in self.df["Charges"]]

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

        assert train_path.exists()
        assert val_path.exists()

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
            [len(eval(x)) == 25 for x in train_df["SequenceEncoding"]]
        ].copy()
        val_df = val_df[[len(eval(x)) == 25 for x in val_df["SequenceEncoding"]]].copy()

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


class iRTDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.encodings = [torch.Tensor(eval(x)).long().T for x in self.df["Encoding"]]
        self.norm_irts = [torch.Tensor([x / 100]).float() for x in self.df["mIRT"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # encoded_pept = torch.Tensor(eval(self.df.iloc[index].Encoding)).long().T
        # norm_irt = torch.Tensor([self.df.iloc[index].mIRT / 100]).float()
        encoded_pept = self.encodings[index]
        norm_irt = self.norm_irts[index]
        return encoded_pept, norm_irt


class iRTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        df = pd.read_csv("/content/summarized_irt_times.csv")

        print(f"Removing Large sequences, currently {len(df)}")
        df = df[[len(eval(x)) == 25 for x in df["Encoding"]]].copy()
        print(f"Left {len(df)}")

        self.df = df

    def setup(self, fold=0, seed=42):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for i, res in enumerate(kf.split(self.df)):
            if i == fold:
                break

        train = self.df.iloc[res[0]].copy()
        val = self.df.iloc[res[1]].copy()

        self.train_dataset = iRTDataset(train)
        self.val_dataset = iRTDataset(val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, num_workers=0, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
