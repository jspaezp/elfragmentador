from __future__ import annotations

from typing import Iterable, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger as lg_logger
from ms2ml import AnnotatedPeptideSpectrum, Peptide
from ms2ml.landmarks import IRT_PEPTIDES
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from elfragmentador.config import CONFIG
from elfragmentador.data.converter import Tensorizer
from elfragmentador.named_batches import ForwardBatch, NamedTensorBatch, TrainBatch

MAX_LENGTH = max(CONFIG.peptide_length_range)


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

    above_max = np.array([len(x) for x in df["seq"]]) > MAX_LENGTH
    lg_logger.info(
        f"Removing {above_max.sum()}/{len(df)} peptides above max length in {path}"
    )
    df = df[~above_max]

    for col in TrainBatch._fields:
        if col in ["seq", "mods"]:
            x = df[col].array
            x = [np.pad(y, pad_width=(0, MAX_LENGTH - len(y))) for y in x]
            x = np.stack(x)
            fields[col] = x
        else:
            fields[col] = np.stack(df[col].array)

    return TrainBatch(**fields)


BatchList = Union[list[TrainBatch], list[ForwardBatch]]


def concat_batches(batches: BatchList) -> NamedTensorBatch:
    elem_type = type(batches[0])
    out = elem_type(*(np.concatenate(samples) for samples in zip(*batches)))
    return out


SplitSet = Literal["Train", "Test", "Val"]

HASHDICT = {
    "__missing__": 0,  # Added manually
    "A": 8990350376580739186,
    "C": -5648131828304525110,
    "D": 6043088297348140225,
    "E": 2424930106316864185,
    "F": 7046537624574876942,
    "G": 3340710540999258202,
    "H": 6743161139278114243,
    "I": -3034276714411840744,
    "K": -6360745720327592128,
    "L": -5980349674681488316,
    "M": -5782039407703521972,
    "N": -5469935875943994788,
    "P": -9131389159066742055,
    "Q": -3988780601193558504,
    "R": -961126793936120965,
    "S": 8601576106333056321,
    "T": -826347925826021181,
    "V": 6418718798924587169,
    "W": -3331112299842267173,
    "X": -7457703884378074688,
    "Y": 2606728663468607544,
    "c_term": 2051117526323448742,
    "n_term": 5536535514417012570,
}


def select_split(pep: Peptide | AnnotatedPeptideSpectrum | str) -> SplitSet:
    """Assigns a peptide to a split set based on its sequence

    It selects all iRT peptides to the 'Val' set.
    The rest of the peptides are hashed based on their stripped sequence (no mods).
    It is done on a semi-random basis

    This function does not strup the sequences, so if passing a string make sure it does
    not have them.

    Args:
        pep (Peptide | AnnotatedPeptideSpectrum | str): Peptide to assign to a split set

    Returns:
        SplitSet: Split set to assign the peptide to.
            This is either one of "Train", "Test" or "Val"

    Examples:
        >>> select_split("AAA")
        'Train'
        >>> select_split("AAAK")
        'Test'
        >>> select_split("AAAKK")
        'Train'
        >>> select_split("AAAMTKK")
        'Train'

    """
    if isinstance(pep, AnnotatedPeptideSpectrum):
        pep = pep.precursor_peptide

    if isinstance(pep, Peptide):
        pep = pep.stripped_sequence

    # Generated using {x:hash(x) for x in CONFIG.encoding_aa_order}
    num_hash = sum(HASHDICT[x] for x in pep)
    return _select_split(pep, num_hash)


def _select_split(pep: str, num_hash: int):
    in_landmark = pep in IRT_PEPTIDES
    number = num_hash / 1e4
    number = number % 1
    assert 0 <= number <= 1

    if number > 0.8 or in_landmark:
        return "Val"
    elif number > 0.6:
        return "Test"
    else:
        return "Train"


def _split_tuple(
    batches_tuple: ForwardBatch | TrainBatch,
) -> dict[SplitSet, ForwardBatch | TrainBatch]:
    aa_arr = np.array(CONFIG.encoding_aa_order)
    hash_arr = np.array(list(HASHDICT.values()))

    def fast_split(x):
        x = x[x > 0]
        pep = aa_arr[x]
        pep = "".join(pep)
        pephash = np.sum(hash_arr[x])
        return _select_split(pep, pephash)

    tuple_type = type(batches_tuple)
    assigned_set = np.array([fast_split(x) for x in tqdm(batches_tuple.seq)])
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
        tensor_tuple = concat_batches(tmp)
        super().__init__(tensor_tuple)
