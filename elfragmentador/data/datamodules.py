from __future__ import annotations

import warnings
from argparse import _ArgumentGroup
from collections import defaultdict
from pathlib import Path, PosixPath

import numpy as np
import pytorch_lightning as pl
import torch
import uniplot
from loguru import logger as lg_logger
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from elfragmentador.config import CONFIG
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
        batches = None
        for x in tqdm(paths):
            tmp = read_cached_parquet(x)
            if batches is None:
                batches = tmp
            else:
                batches = concat_batches([batches, tmp])

        lg_logger.info("Splitting train/test/val")

        self.batches = _split_tuple(batches)
        self.len_train = len(self.batches["Train"][0])
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
        glimpse_tensor_tuple(tmp)
        shuffle = subset == "Train"
        out = TupleTensorDataset(tmp).as_dataloader(
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
        return out


def glimpse_tensor_tuple(tb: TrainBatch):
    length = len(tb.seq)
    lg_logger.info(f"Number of sequences: {length}")
    seq_mask = tb.seq != 0

    mean_length = seq_mask.sum(axis=1).float().mean()
    lg_logger.info(f"Mean length: {mean_length}")

    mod_mask = tb.mods != 0
    uniq_counts = defaultdict(int)

    for x, y in zip(tb.seq[mod_mask].flatten(), tb.mods[mod_mask].flatten()):
        uniq_counts[(int(x), int(y))] += 1

    aas = CONFIG.encoding_aa_order
    mods = CONFIG.encoding_mod_order
    uniq_counts_translated = {
        f"{aas[k[0]]}{mods[k[1]]}": v for k, v in uniq_counts.items() if v > 1
    }

    lg_logger.info(f"AA modifications in the dataset: {uniq_counts_translated}")

    spec_mask = tb.spectra > 0
    mean_specs = spec_mask.sum(axis=1).float().mean()
    lg_logger.info(f"Mean number of peaks per spectra: {mean_specs}")

    uniplot.histogram(
        np.sqrt(tb.spectra[spec_mask].numpy().flatten()),
        bins=100,
        title="sqrt of non-0 intensities of peaks",
    )

    nce_counts = np.unique(tb.nce.numpy(), return_counts=True)
    lg_logger.info(f"Number of spectra per NCE: {dict(zip(*nce_counts))}")

    tot_missing_irt = tb.irt.isnan().sum()
    avg_nonmissing_irt = tb.irt[~tb.irt.isnan()].mean()
    max_nonmissing_irt = tb.irt[~tb.irt.isnan()].max()
    min_nonmissing_irt = tb.irt[~tb.irt.isnan()].min()
    lg_logger.info(f"Missing iRT values: {tot_missing_irt}/{length}")
    lg_logger.info(f"Average iRT value: {avg_nonmissing_irt}")
    lg_logger.info(f"Max iRT value: {max_nonmissing_irt}")
    lg_logger.info(f"Min iRT value: {min_nonmissing_irt}")
