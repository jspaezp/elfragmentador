from typing import Iterable, Literal

import numpy as np
import pandas as pd
from loguru import logger as lg_logger
from ms2ml import Peptide
from ms2ml.data.adapters import BaseAdapter
from torch.utils.data import DataLoader, TensorDataset

from elfragmentador.config import get_default_config
from elfragmentador.data.converter import Tensorizer
from elfragmentador.named_batches import ForwardBatch, NamedTensorBatch, TrainBatch

DEFAULT_CONFIG = get_default_config()
MAX_LENGTH = max(DEFAULT_CONFIG.peptide_length_range)


class TupleTensorDataset(TensorDataset):
    def __init__(self, tensor_tuple):
        super().__init__(*tensor_tuple)
        self.builder = type(tensor_tuple)

    def __getitem__(self, index):
        out = self.builder(*super().__getitem__(index))
        return out

    def as_dataloader(self, batch_size, shuffle, num_workers=0, *args, **kwargs):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs,
        )


def read_cached_parquet(path) -> TrainBatch:
    df = pd.read_parquet(path=path)

    fields = {}

    for col in TrainBatch._fields:
        if col in ["seq", "mods"]:
            x = df[col].array
            x = [np.pad(y, pad_width=(0, MAX_LENGTH - len(y))) for y in x]
            x = np.stack(x)
            fields[col] = x
        else:
            fields[col] = np.stack(df[col].array)

    return TrainBatch(**fields)


BatchList = list[TrainBatch] | list[ForwardBatch]


def concat_batches(batches: BatchList) -> NamedTensorBatch:
    elem_type = type(batches[0])
    out = elem_type(*(np.concatenate(samples) for samples in zip(*batches)))
    return out


SplitSet = Literal["Train", "Test", "Val"]


def _select_split(pep: Peptide) -> SplitSet:
    num_hash = hash(pep.stripped_sequence)
    lg_logger.debug(f"{pep.stripped_sequence()}: {num_hash}")
    number = num_hash / 1e4
    number = number % 1

    if number > 0.8:
        return "Val"
    elif number > 0.6:
        return "Test"
    else:
        return "Train"


def _split_tuple(
    batches_tuple: ForwardBatch | TrainBatch,
) -> dict[SplitSet, ForwardBatch | TrainBatch]:
    def pep_builder(x):
        x = x[x > 0]
        return Peptide.decode_vector(seq=x, mod=np.zeros_like(x), config=DEFAULT_CONFIG)

    tuple_type = type(batches_tuple)
    assigned_set = np.array([_select_split(pep_builder(x)) for x in batches_tuple.seq])
    counts = np.unique(assigned_set, return_counts=True)
    lg_logger.info(f"Splitting dataset into train/test/val groups: {counts}")

    out = {}
    for subset in ["Train", "Test", "Val"]:
        out[subset] = tuple_type(*[x[assigned_set == subset] for x in batches_tuple])

    return out


class PeptideDataset(TupleTensorDataset):
    def __init__(self, peptide_list: Iterable[Peptide], nce=None, charge=None):
        converter = Tensorizer(nce=nce)
        tmp = [converter(x) for x in peptide_list]
        tensor_tuple = combine_batches(tmp)
        super().__init__(tensor_tuple)
