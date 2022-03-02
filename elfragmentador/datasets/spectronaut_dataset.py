from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from elfragmentador.datasets.batch_utils import _append_batch_to_df
from elfragmentador.datasets.dataset import DatasetBase
from elfragmentador.named_batches import TrainBatch
from elfragmentador.spectra import Spectrum


class SpectronautLibrary(DatasetBase):
    def __init__(self, in_path: PathLike, nce: float = 0, df=None) -> None:
        super().__init__()
        self.in_path = in_path
        if df is None:
            self.df = pd.read_csv(in_path)
        self.overwrite_nce = nce
        self.spectra = []

        for i, x in self.df.groupby(
            ["ModifiedPeptide", "PrecursorCharge", "PrecursorMz"]
        ):
            self.spectra.append(
                Spectrum(
                    i[0],
                    charge=i[1],
                    parent_mz=i[2],
                    mzs=x["FragmentMz"],
                    intensities=x["RelativeIntensity"],
                    irt=np.unique(x["iRT"]),
                    nce=0,
                )
            )

    def __getitem__(self, index):
        if hasattr(self, "greedy_cache"):
            return self.__greedy_getitem__(index)
        else:
            return self.__lazy_getitem__(index)

    def __lazy_getitem__(self, index):
        foo = self.spectra[index]

        spec = foo.encode_spectra()
        seqs = foo.encode_sequence(enforce_length=False, pad_zeros=False)
        charge = foo.charge
        nce = self.calc_nce(foo.nce)
        irt = foo.irt / 100

        out = TrainBatch(
            seq=torch.tensor(seqs.aas),
            mods=torch.tensor(seqs.mods),
            nce=torch.tensor(nce),
            charge=torch.tensor(charge),
            spectra=torch.tensor(spec),
            irt=torch.tensor(irt),
            weight=torch.tensor(float("nan")),
        )

        return out

    def __greedy_getitem__(self, index):
        if hasattr(self, "greedy_cache"):
            return self.greedy_cache[index]
        else:
            raise RuntimeError(
                "Attempted to get item from greedy cache, "
                "please call self.greedyfy() before"
            )

    def __len__(self):
        return len(self.spectra)

    def greedify(self):
        self.greedy_cache = [self.__lazy_getitem__(i) for i in range(len(self))]

    def top_n_subset(self, n) -> SpectronautLibrary:
        out = SpectronautLibrary(
            in_path=self.in_path, df=self.df.head(n), nce=self.overwrite_nce
        )
        return out

    def append_batches(self, batches: NamedTuple, prefix: str):
        _append_batch_to_df(df=self.df, batches=batches, prefix=prefix)

    def save_data(self, prefix: PathLike):
        path = Path(prefix)
        csv_path = str(path) + ".csv"
        feather_path = str(path) + ".feather"
        logging.info(f"Saving data from {self} to {csv_path} and {feather_path}")
        self.df.to_csv(csv_path, index=False)
        self.df.reset_index(drop=True).to_feather(feather_path)
