from __future__ import annotations

from typing import Optional
from os import PathLike

import pandas as pd
from pandas import DataFrame

from elfragmentador.utils import torch_batch_from_seq
from elfragmentador.spectra import Spectrum
from elfragmentador.datasets.dataset import DatasetBase, Predictor
from elfragmentador.named_batches import ForwardBatch, PredictionResults
from elfragmentador.model import PepTransformerModel


class SequenceDataset(DatasetBase):
    def __init__(self, sequences, collision_energies, charges) -> None:
        super().__init__()
        self.sequences, self.collision_energies, self.charges = (
            sequences,
            collision_energies,
            charges,
        )
        self.batches = []
        for s, n, c in zip(sequences, collision_energies, charges):
            tmp = torch_batch_from_seq(seq=s, nce=n, charge=c)
            out = ForwardBatch(**{k: x.squeeze(0) for k, x in tmp._asdict().items()})
            self.batches.append(out)
        self.predictions = None

    @staticmethod
    def from_csv(path: PathLike):
        return SequenceDataset.from_df(pd.read_csv(path))

    @staticmethod
    def from_df(df: DataFrame):
        OPTION_1_NAMES = ["Modified Sequence", "CE", "Precursor Charge"]
        OPTION_2_NAMES = ["modified_sequence", "collision_energy", "precursor_charge"]

        if OPTION_1_NAMES[0] in list(df):
            names = OPTION_1_NAMES
        elif OPTION_2_NAMES[0] in list(df):
            names = OPTION_2_NAMES
        else:
            raise ValueError(
                "Names in the data frame dont match any of the posible options"
            )

        sequences = df[names[0]]
        nces = df[names[1]]
        charges = df[names[2]]

        dataset = SequenceDataset(
            sequences=sequences, charges=charges, collision_energies=nces
        )
        return dataset

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)

    @staticmethod
    def convert_to_spectrum(in_batch: ForwardBatch, out: PredictionResults) -> str:
        out = PredictionResults(**{k: x.squeeze(0) for k, x in out._asdict().items()})

        # rt should be in seconds for spectrast ...
        # irt should be non-dimensional

        out = Spectrum.from_tensors(
            sequence_tensor=in_batch.seq.squeeze().numpy(),
            fragment_tensor=out.spectra / out.spectra.max(),
            mod_tensor=in_batch.mods.squeeze().numpy(),
            charge=int(in_batch.charge),
            nce=float(in_batch.nce),
            rt=float(out.irt) * 100 * 60,
            irt=float(out.irt) * 100,
        )

        return out

    def generate_sptxt(self, outfile: PathLike):
        if self.predicted_irt is None:
            raise ValueError(
                "No predictions found, run 'SequenceDataset.predict' first"
            )

        with open(outfile, "w") as f:
            for ib, irt, spec in zip(
                self.batches, self.predicted_irt, self.predicted_spectra
            ):
                ob = PredictionResults(irt=irt, spectra=spec)
                spec = self.convert_to_spectrum(ib, ob).to_sptxt()
                f.write(spec + "\n")

    def predict(
        self,
        model: PepTransformerModel,
        predictor: Optional[Predictor] = None,
        batch_size: int = 4,
    ):
        if predictor is None:
            predictor = Predictor(batch_size=batch_size)

        predictions = predictor.predict_dataset(
            model=model,
            dataset=self,
        )
        self.predicted_irt = predictions.irt
        self.predicted_spectra = predictions.spectra

    def append_batches(self, batches):
        self.cached_batches = batches

    def save_data(self, prefix: PathLike):
        return self.generate_sptxt(prefix + ".sptxt")

    def top_n_subset(self, n):
        raise ValueError(
            "Top N is not relevant in the context of a dataset without ground truth"
        )

    def greedify(self):
        pass
