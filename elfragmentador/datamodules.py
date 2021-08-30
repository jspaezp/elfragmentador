from __future__ import annotations

import logging

import warnings
from collections import namedtuple
from pathlib import PosixPath, Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import torch
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl

from elfragmentador import constants, spectra
from elfragmentador.encoding_decoding import decode_mod_seq
from argparse import _ArgumentGroup
from torch import Tensor
from tqdm.auto import tqdm

TrainBatch = namedtuple(
    "TrainBatch",
    "encoded_sequence, encoded_mods, charge, nce, encoded_spectra, norm_irt",
)


def convert_tensor_column(column, elem_function=float, *args, **kwargs):
    """converts a series (column in a pandas dataframe) to a tensor

    Expects a column whose character values actually represent lists of numbers
    for exmaple "[1,2,3,4]"

    Args:
        column: Series/list of stirngs
        elem_function: function to use to convert each substring in a value (Default value = float)
        *args: Arguments to pass to tqdm
        **kwargs: Keywords arguments passed to tqdm

    Returns:
        List: A nested list containing the converted values

    Examples:
        >>> convert_tensor_column(["[2,2]", "[3,3]"], elem_function=float)
        [array([2., 2.]), array([3., 3.])]
    """

    if isinstance(column[0], str):
        out = [
            np.array(
                [
                    elem_function(y.strip())
                    for y in x.strip("[]").replace("'", "").split(",")
                ]
            )
            for x in tqdm(column, *args, **kwargs)
        ]

    elif isinstance(column[0], list) or isinstance(column[0], tuple):
        out = [
            np.array([elem_function(y) for y in x])
            for x in tqdm(column, *args, **kwargs)
        ]

    else:
        logging.warning("Passed column is not a string, skipping conversion")
        out = column

    return out


def match_lengths(
    nested_list: Union[List[List[Union[int, float]]], List[List[int]]],
    max_len: int,
    name: str = "items",
) -> Tensor:
    """
    match_lengths Matches the lengths of all tensors in a list

    Args:
        nested_list (List[np.ndarray]):
            A list of numpy arrays
        max_len (int): Length to match all tensors to
        name (str, optional): name to use (just for logging purposes). Defaults to "items".

    Returns:
        Tensor:
            Tensor product of stacking all the elements in the input nested list, after
            equalizing the length of all of them to the specified max_len

    Examples:
        >>> out = match_lengths([np.array([3, 3]), np.array([1, 2])], 2, "")
        >>> out
        tensor([[3, 3],
                [1, 2]])
        >>> out[0]
        tensor([3, 3])
        >>> match_lengths([np.array([1]), np.array([1, 2])], 2, "")
        tensor([[1, 0],
                [1, 2]])
    """
    lengths = np.array([len(x) for x in nested_list])
    unique_lengths = set(lengths)
    match_max = lengths == max_len

    out_message = (
        f"{len(match_max)}/{len(nested_list)} "
        f"{name} actually match the max sequence length of"
        f" {max_len},"
        f" found {unique_lengths}"
    )

    if (len(unique_lengths) == 1) and (max_len in unique_lengths):
        logging.info(out_message)
        out = torch.stack([torch.from_numpy(x) for x in nested_list], dim=0)
    else:
        logging.warning(out_message)
        out = [
            np.concatenate([x, np.zeros(max_len - len(x), dtype=x.dtype)])
            if len(x) != max_len
            else x
            for x in nested_list
        ]
        out = torch.stack([torch.from_numpy(x) for x in out], dim=0)

    return out


def match_colnames(df: DataFrame) -> Dict[str, Optional[str]]:
    """
    match_colnames Tries to find aliases for columns names in a data frame


    Tries to find the following column aliases:

    "SeqE": Sequence encoding
    "ModE": Modification Encoding
    "SpecE": Spectrum encoding
    "Ch": Charge
    "iRT": Retention time
    "NCE": Collision Energy


    Args:
        df (DataFrame): Data frame to find columns for ...

    Returns:
        Dict[str, Optional[str]]:
            Dictionary with the aliases (keys are the ones specified in the details section)
    """

    def _match_col(string1, string2, colnames, match_mode="in", combine_mode=None):
        m = {
            "in": lambda q, t: q in t,
            "startswith": lambda q, t: q.startswith(t) or t.startswith(q),
            "equals": lambda q, t: q == t,
        }
        match_fun = m[match_mode]
        match_indices1 = [i for i, x in enumerate(colnames) if match_fun(string1, x)]

        if string2 is None:
            match_indices = match_indices1
        else:
            match_indices2 = [
                i for i, x in enumerate(colnames) if match_fun(string2, x)
            ]
            if combine_mode == "union":
                match_indices = set(match_indices1).union(set(match_indices2))
            elif combine_mode == "intersect":
                match_indices = set(match_indices1).intersection(set(match_indices2))
            else:
                raise NotImplementedError

        try:
            out_index = list(match_indices)[0]
        except IndexError:
            out_index = None

        return out_index

    colnames = list(df)
    out = {
        "SeqE": _match_col("Encoding", "Seq", colnames, combine_mode="intersect"),
        "ModE": _match_col("Encoding", "Mod", colnames, combine_mode="intersect"),
        "SpecE": _match_col("Encoding", "Spec", colnames, combine_mode="intersect"),
        "Ch": _match_col("harg", None, colnames),
        "iRT": _match_col("IRT", "iRT", colnames, combine_mode="union"),
        "NCE": _match_col(
            "nce", "NCE", colnames, combine_mode="union", match_mode="startswith"
        ),
    }
    out = {k: (colnames[v] if v is not None else None) for k, v in out.items()}
    logging.info(f">>> Mapped column names to the provided dataset {out}")
    return out


class PeptideDataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(
        self,
        df: DataFrame,
        max_spec: int = 2e6,
        drop_missing_vals=False,
    ) -> None:
        super().__init__()
        logging.info(">>> Initalizing Dataset")
        if drop_missing_vals:
            former_len = len(df)
            df.dropna(inplace=True)
            logging.warning(
                f"\n>>> {former_len}/{len(df)} rows left after dropping missing values"
            )

        if max_spec < len(df):
            logging.warning(
                ">>> Filtering out to have "
                f"{max_spec}, change the 'max_spec' argument if you don't want"
                "this to happen"
            )
            df = df.sample(n=int(max_spec))

        self.df = df  # TODO remove this for memory ...

        name_match = match_colnames(df)

        sequence_encodings = convert_tensor_column(
            self.df[name_match["SeqE"]], int, "Decoding sequence encodings"
        )
        sequence_encodings = match_lengths(
            sequence_encodings, constants.MAX_TENSOR_SEQUENCE, "Sequences"
        )
        self.sequence_encodings = sequence_encodings.long()

        mod_encodings = convert_tensor_column(
            self.df[name_match["ModE"]], int, "Decoding Modification encoding"
        )
        mod_encodings = match_lengths(
            mod_encodings, constants.MAX_TENSOR_SEQUENCE, "Mods"
        )
        self.mod_encodings = mod_encodings.long()

        spectra_encodings = convert_tensor_column(
            self.df[name_match["SpecE"]], float, "Decoding Spec Encodings"
        )
        spectra_encodings = match_lengths(
            spectra_encodings, constants.NUM_FRAG_EMBEDINGS, "Spectra"
        )
        self.spectra_encodings = spectra_encodings.float()
        avg_peaks = torch.sum(spectra_encodings > 0.01, axis=1).float().mean()

        spectra_lengths = len(self.spectra_encodings[0])
        sequence_lengths = len(self.sequence_encodings[0])

        try:
            irts = np.array(self.df[name_match["iRT"]]).astype("float") / 100
            self.norm_irts = torch.from_numpy(irts).float().unsqueeze(1)
            del irts
        except ValueError as e:
            logging.error(self.df[name_match["iRT"]])
            logging.error(e)
            raise e

        if name_match["NCE"] is None:
            nces = (
                torch.Tensor([float("nan")] * len(self.norm_irts)).float().unsqueeze(1)
            )
        else:
            try:
                nces = np.array(self.df[name_match["NCE"]]).astype("float")
                nces = torch.from_numpy(nces).float().unsqueeze(1)
            except ValueError as e:
                logging.error(self.df[name_match["NCE"]])
                logging.error(e)
                raise e

        self.nces = nces

        if torch.any(self.nces.isnan()):
            # TODO decide if here should be the place to impute NCEs ... and warn ...
            warnings.warn(
                (
                    "Found missing values in NCEs, assuming 30."
                    " Please fix the data for future use, "
                    "since this imputation will be removed in the future"
                ),
                FutureWarning,
            )
            self.nces = torch.where(self.nces.isnan(), torch.Tensor([30.0]), self.nces)

            # This syntax is compatible in torch +1.8, will change when colab migrates to it
            # self.nces = torch.nan_to_num(self.nces, nan=30.0)

        charges = np.array(self.df[name_match["Ch"]]).astype("long")
        self.charges = torch.Tensor(charges).long().unsqueeze(1)

        logging.info(
            (
                f"Dataset Initialized with {len(df)} entries."
                f" Sequence length: {sequence_lengths}"
                f" Spectra length: {spectra_lengths}"
                f"; Average Peaks/spec: {avg_peaks}"
            )
        )
        logging.info(">>> Done Initializing dataset\n")
        del self.df

    @property
    def mod_sequences(self):
        """ """
        if not hasattr(self, "_mod_sequences"):
            self._mod_sequences = [
                decode_mod_seq([int(s) for s in seq], [int(m) for m in mod])
                for seq, mod in zip(self.sequence_encodings, self.mod_encodings)
            ]

        return self._mod_sequences

    @staticmethod
    def from_sptxt(
        filepath: str,
        max_spec: int = 1e6,
        filter_df: bool = True,
        *args,
        **kwargs,
    ) -> PeptideDataset:
        df = spectra.SptxtReader(str(filepath), *args, **kwargs).to_df(
            max_spec=max_spec
        )
        if filter_df:
            df = filter_df_on_sequences(df)

        return PeptideDataset(df)

    @staticmethod
    def from_csv(filepath: Union[str, Path], max_spec: int = 1e6):
        df = filter_df_on_sequences(pd.read_csv(str(filepath)))
        return PeptideDataset(df, max_spec=max_spec)

    def __len__(self) -> int:
        return len(self.sequence_encodings)

    def __getitem__(self, index: int) -> TrainBatch:
        encoded_sequence = self.sequence_encodings[index]
        encoded_mods = self.mod_encodings[index]
        encoded_spectra = self.spectra_encodings[index]
        norm_irt = self.norm_irts[index]
        charge = self.charges[index]
        nce = self.nces[index]

        out = TrainBatch(
            encoded_sequence=encoded_sequence,
            encoded_mods=encoded_mods,
            charge=charge,
            nce=nce,
            encoded_spectra=encoded_spectra,
            norm_irt=norm_irt,
        )
        return out


def filter_df_on_sequences(df: DataFrame, name: str = "") -> DataFrame:
    """
    filter_df_on_sequences Filters a ataframe for the peptides that correctly match the expected lengths

    Args:
        df (DataFrame): A DataFrame to filter
        name (str, optional): Only used for debugging purposes. Defaults to "".

    Returns:
        DataFrame: The filtered dataframe
    """
    name_match = match_colnames(df)
    logging.info(list(df))
    logging.warning(f"Removing Large sequences, currently {name}: {len(df)}")

    seq_iterable = convert_tensor_column(
        df[name_match["SeqE"]], int, "Decoding tensor seqs"
    )
    df[name_match["SeqE"]] = seq_iterable
    seq_len_matching = [len(x) <= constants.MAX_TENSOR_SEQUENCE for x in seq_iterable]
    if sum(seq_len_matching) == 0:
        warnings.warn("No sequences have the expected length")

    df = df[seq_len_matching].copy().reset_index(drop=True)

    logging.warning(f"Left {name}: {len(df)}")
    return df


class PeptideDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        base_dir: Union[str, PosixPath] = ".",
        drop_missing_vals: bool = False,
        max_spec = 2_000_000,
    ) -> None:
        super().__init__()
        logging.info("Initializing DataModule")
        self.batch_size = batch_size
        self.drop_missing_vals = drop_missing_vals
        self.max_spec = max_spec
        base_dir = Path(base_dir)

        train_path = list(base_dir.glob("*train*.csv*"))
        val_path = list(base_dir.glob("*val*.csv*"))

        assert len(train_path) > 0, f"Train File not found in '{base_dir}'"
        assert len(val_path) > 0, f"Val File not found in '{base_dir}'"

        logging.info("Starting loading of the data")
        train_df = pd.concat([pd.read_csv(str(x)) for x in train_path])
        val_df = pd.concat([pd.read_csv(str(x)) for x in val_path])

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
            self.train_df, drop_missing_vals=self.drop_missing_vals, max_spec=self.max_spec
        )
        self.val_dataset = PeptideDataset(
            self.val_df, drop_missing_vals=self.drop_missing_vals, max_spec=self.max_spec
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, num_workers=0, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, num_workers=0, batch_size=self.batch_size)
