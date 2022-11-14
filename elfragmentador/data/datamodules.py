from __future__ import annotations

import warnings
from argparse import _ArgumentGroup
from pathlib import Path, PosixPath

import pytorch_lightning as pl
import torch
from loguru import logger as lg_logger
from torch.utils.data.dataloader import DataLoader

from elfragmentador.data.torch_datasets import (
    SplitSet,
    TupleTensorDataset,
    _split_tuple,
    concat_batches,
    read_cached_parquet,
)
from elfragmentador.named_batches import TrainBatch


class TrainingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        base_dir: str | PosixPath = ".",
    ) -> None:
        super().__init__()
        lg_logger.info("Initializing DataModule")
        base_dir = Path(base_dir)
        paths = list(base_dir.rglob("*.parquet"))

        lg_logger.info(f"Found {len(paths)} parquet files: {paths}")
        batches = concat_batches([read_cached_parquet(x) for x in paths])

        lg_logger.info("Splitting train/test/val")

        self.batches = _split_tuple(batches)
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parser: _ArgumentGroup) -> _ArgumentGroup:
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_dir", type=str, default=".")
        return parser

    def train_dataloader(self) -> DataLoader:
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )
        return self.get_batch_dataloader("Train")

    def val_dataloader(self) -> DataLoader:
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )
        return self.get_batch_dataloader("Val")

    def test_dataloader(self) -> DataLoader:
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )
        return self.get_batch_dataloader("Test")

    def get_batch_set(self, subset: SplitSet = "Train"):
        tmp = self.batches[subset]
        out = TrainBatch(*[torch.from_numpy(x) for x in tmp])
        return out

    def get_batch_dataloader(self, subset: SplitSet = "Train"):
        tmp = self.get_batch_set(subset=subset)
        shuffle = subset == "Train"
        out = TupleTensorDataset(tmp).as_dataloader(
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
        return out
