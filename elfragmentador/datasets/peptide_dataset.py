from __future__ import annotations

import logging
import warnings
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader

from elfragmentador import constants, spectra
from elfragmentador.datasets.dataset import DatasetBase
from elfragmentador.encoding_decoding import decode_mod_seq
from elfragmentador.named_batches import TrainBatch
from elfragmentador.utils_data import (
    _convert_tensor_columns_df,
    _filter_df_on_sequences,
    _match_colnames,
    _match_lengths,
)


class PeptideDataset(DatasetBase):
    @torch.no_grad()
    def __init__(
        self,
        df: DataFrame,
        max_spec: int = 2e6,
        drop_missing_vals=False,
        filter_df: bool = False,
        keep_df: bool = False,
    ) -> None:
        super().__init__()
        logging.info(">>> Initalizing Dataset")

        if filter_df:
            df = _filter_df_on_sequences(df)

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

        self.df = df

        df = _convert_tensor_columns_df(df)
        name_match = _match_colnames(df)

        self.sequence_encodings = _match_lengths(
            self.df[name_match["SeqE"]],
            constants.MAX_TENSOR_SEQUENCE,
            "Sequences",
        ).long()

        self.mod_encodings = _match_lengths(
            self.df[name_match["ModE"]], constants.MAX_TENSOR_SEQUENCE, "Mods"
        ).long()

        self.spectra_encodings = _match_lengths(
            self.df[name_match["SpecE"]], constants.NUM_FRAG_EMBEDINGS, "Spectra"
        ).float()

        avg_peaks = torch.sum(self.spectra_encodings > 0.001, axis=1).float().mean()
        spectra_lengths = len(self.spectra_encodings[0])
        sequence_lengths = len(self.sequence_encodings[0])

        irts = torch.from_numpy(np.array(self.df[name_match["iRT"]]).astype("float"))
        if torch.all(torch.isnan(irts)):
            irts = torch.from_numpy(np.array(self.df[name_match["RT"]]).astype("float"))

        self.norm_irts = (irts / 100).float().unsqueeze(1)
        self.nces = (
            torch.from_numpy(np.array(self.df[name_match["NCE"]]).astype("float"))
            .float()
            .unsqueeze(1)
        )

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

            # This syntax is compatible in torch +1.8,
            # will change when colab migrates to it
            # self.nces = torch.nan_to_num(self.nces, nan=30.0)

        self.charges = (
            torch.from_numpy(np.array(self.df[name_match["Ch"]]).astype("long"))
            .long()
            .unsqueeze(1)
        )
        self.weights = (
            torch.from_numpy(np.array(self.df[name_match["Weight"]]).astype("float"))
            .long()
            .unsqueeze(1)
        )
        self.weights = torch.sqrt(self.weights)

        logging.info(
            (
                f"Dataset Initialized with {len(df)} entries."
                f" Sequence length: {sequence_lengths}"
                f" Spectra length: {spectra_lengths}"
                f"; Average Peaks/spec: {avg_peaks}"
            )
        )
        logging.info(">>> Done Initializing dataset\n")

        if not keep_df:
            del self.df

    @property
    def mod_sequences(self):
        """
        Returns the mod sequences as a list of strings
        """
        if not hasattr(self, "_mod_sequences"):
            self._mod_sequences = []

            for seq, mod in zip(self.sequence_encodings, self.mod_encodings):
                mod = F.pad(mod, (0, seq.size(-1)), mode="constant")
                seq = [int(s) for s in seq]
                mods = [int(m) for m in mod]
                seq = decode_mod_seq(seq, mods)
                self._mod_sequences.append(seq)

        return self._mod_sequences

    @staticmethod
    def from_sptxt(
        filepath: str,
        max_spec: int = 1e6,
        filter_df: bool = True,
        keep_df: bool = False,
        min_peaks: int = 3,
        min_delta_ascore: int = 20,
        enforce_length=True,
        pad_zeros=True,
        *args,
        **kwargs,
    ) -> PeptideDataset:
        df = spectra.SptxtReader(str(filepath), *args, **kwargs).to_df(
            max_spec=max_spec,
            min_peaks=min_peaks,
            min_delta_ascore=min_delta_ascore,
            enforce_length=enforce_length,
            pad_zeros=pad_zeros,
        )
        if filter_df:
            df = _filter_df_on_sequences(df)

        return PeptideDataset(df, keep_df=keep_df)

    @staticmethod
    def from_csv(
        filepath: Union[str, Path],
        max_spec: int = 1e6,
        filter_df: bool = True,
        keep_df: bool = False,
    ):
        df = pd.read_csv(str(filepath))
        if filter_df:
            df = _filter_df_on_sequences(df)
        return PeptideDataset(
            df, max_spec=max_spec, filter_df=filter_df, keep_df=keep_df
        )

    @staticmethod
    def from_feather(
        filepath: PathLike,
        max_spec: int = 2e6,
        filter_df: bool = True,
        keep_df: bool = False,
    ):
        df = pd.read_feather(str(filepath))
        if filter_df:
            df = _filter_df_on_sequences(df)
        return PeptideDataset(df, max_spec=max_spec, keep_df=keep_df)

    def as_dataloader(self, batch_size, shuffle, num_workers=0, *args, **kwargs):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs,
        )

    def greedify(self):
        pass

    def top_n_subset(self, n, *args, **kwargs):
        logging.info("Ignoring metric when subsetting in peptide dataset")
        if not hasattr(self, "df"):
            msg = "Please generate this dataset again "
            msg += "using the `keep_df = True` argument"

            raise AttributeError(msg)

        df = self.df.sample(min(n, len(self.df)))

        return PeptideDataset(df, max_spec=n, keep_df=True)

    def append_batches(self, batches, prefix=""):
        for k, v in batches._asdict().items():
            self.df.insert(
                loc=len(list(self.df)) - 2, column=prefix + k, value=float("nan")
            )
            self.df[prefix + k] = v

    def save_data(self, prefix: PathLike):
        self.df.reset_index(drop=True).to_csv(prefix + ".csv", index=False)
        self.df.reset_index(drop=True).to_feather(prefix + ".feather")

    def __len__(self) -> int:
        return len(self.sequence_encodings)

    def __getitem__(self, index: int) -> TrainBatch:
        encoded_sequence = self.sequence_encodings[index]
        encoded_mods = self.mod_encodings[index]
        encoded_spectra = self.spectra_encodings[index]
        norm_irt = self.norm_irts[index]
        charge = self.charges[index]
        nce = self.calc_nce(self.nces[index])

        weight = self.weights[index]

        out = TrainBatch(
            seq=encoded_sequence.clone(),
            mods=encoded_mods.clone(),
            charge=charge.clone(),
            nce=nce.clone(),
            spectra=encoded_spectra.clone(),
            irt=norm_irt.clone(),
            weight=weight.clone(),
        )
        return out
