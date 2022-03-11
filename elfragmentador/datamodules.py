from __future__ import annotations

import logging
import warnings
from argparse import _ArgumentGroup
from pathlib import Path, PosixPath
from typing import Union

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from elfragmentador import utils_data
from elfragmentador.datasets.peptide_dataset import PeptideDataset
from elfragmentador.utils_data import _convert_tensor_columns_df


class PeptideDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        base_dir: Union[str, PosixPath] = ".",
        drop_missing_vals: bool = False,
        max_spec=2_000_000,
    ) -> None:
        super().__init__()
        logging.info("Initializing DataModule")
        self.batch_size = batch_size
        self.drop_missing_vals = drop_missing_vals
        self.max_spec = max_spec
        base_dir = Path(base_dir)

        if len(list(base_dir.glob("*.feather"))) > 0:
            reader = pd.read_feather
            glob_str = "feather"
        else:
            reader = pd.read_csv
            glob_str = "csv"

        train_path = list(base_dir.glob(f"*train*.{glob_str}*"))
        val_path = list(base_dir.glob(f"*val*.{glob_str}*"))

        assert (
            len(train_path) > 0
        ), f"Train File not found in '{base_dir}'\nFound {list(base_dir.glob('*'))}"
        assert (
            len(val_path) > 0
        ), f"Val File not found in '{base_dir}'\nFound {list(base_dir.glob('*'))}"

        logging.info("Starting loading of the data")
        train_df = pd.concat(
            [_convert_tensor_columns_df(reader(str(x))) for x in train_path]
        )
        val_df = pd.concat(
            [_convert_tensor_columns_df(reader(str(x))) for x in val_path]
        )

        # TODO reconsider if storing this dataframe as is is required
        self.train_df = train_df
        self.val_df = val_df

    @staticmethod
    def add_model_specific_args(parser: _ArgumentGroup) -> _ArgumentGroup:
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--drop_missing_vals", type=bool, default=False)
        parser.add_argument("--max_spec", type=int, default=2000000)
        return parser

    def setup(self) -> None:
        self.train_dataset = PeptideDataset(
            self.train_df,
            drop_missing_vals=self.drop_missing_vals,
            max_spec=self.max_spec,
        )
        self.val_dataset = PeptideDataset(
            self.val_df,
            drop_missing_vals=self.drop_missing_vals,
            max_spec=self.max_spec,
        )

    def train_dataloader(self) -> DataLoader:
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )
        return self.train_dataset.as_dataloader(
            num_workers=0,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=utils_data.collate_fun,
        )

    def val_dataloader(self) -> DataLoader:
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )
        return self.val_dataset.as_dataloader(
            num_workers=0,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=utils_data.collate_fun,
        )
