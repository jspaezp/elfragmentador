from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pandas import DataFrame
from pyteomics import fasta, parser
from tqdm.auto import tqdm

from elfragmentador.datasets.dataset import DatasetBase, Predictor
from elfragmentador.model import PepTransformerModel
from elfragmentador.named_batches import ForwardBatch, PredictionResults
from elfragmentador.spectra import Spectrum
from elfragmentador.utils import torch_batch_from_seq


class SequenceDataset(DatasetBase):
    def __init__(
        self, sequences: List[str], collision_energies: List[float], charges: List[int]
    ) -> None:
        """
        Dataset that contains sequences to be used to predict spectra.

        Args:
            sequences (List[str]): List of modified peptide sequences
            collision_energies (List[float]): List of collision energies
            charges (List[int]): List of charges

        Examples:
            >>> seqs = ["MYPEPTIDEK", "PEPT[PHOSPHO]IDEPINK"]
            >>> ces, charges = [27, 28], [2, 3]
            >>> my_ds = SequenceDataset(seqs, ces, charges)
            >>> my_ds[0]
            ForwardBatch(seq=tensor([23, 11, 21, 13,  4, 13, 17,  8,  3,  4,  9, 22]), \
                mods=tensor([0, ..., 0]), charge=tensor([2]), nce=tensor([27.]))
        """
        super().__init__()
        self.sequences, self.collision_energies, self.charges = (
            sequences,
            collision_energies,
            charges,
        )
        self.batches = []
        my_iter = tqdm(
            zip(sequences, collision_energies, charges),
            total=len(sequences),
            desc="Generating Tensors",
        )
        for s, n, c in my_iter:
            self.batches.append(self.make_batch_element(seq=s, nce=n, charge=c))
        self.predictions = None

    @staticmethod
    def make_batch_element(seq, nce, charge):
        tmp = torch_batch_from_seq(
            seq=seq, nce=nce, charge=charge, enforce_length=False, pad_zeros=False
        )
        out = ForwardBatch(**{k: x.squeeze(0) for k, x in tmp._asdict().items()})
        return out

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
            rt=float(out.irt) * 60,
            irt=float(out.irt),
        )

        return out

    def generate_sptxt(self, outfile: PathLike):
        if self.predicted_irt is None:
            raise ValueError(
                "No predictions found, run 'SequenceDataset.predict' first"
            )

        my_iter = tqdm(
            zip(self.batches, self.predicted_irt, self.predicted_spectra),
            desc=f"Writting Spectra to {outfile}",
            total=len(self.predicted_spectra),
        )

        with open(outfile, "w") as f:
            for ib, irt, spec in my_iter:
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


class FastaDataset(SequenceDataset):
    def __init__(
        self,
        fasta_file: PathLike,
        enzyme="trypsin",
        missed_cleavages=2,
        min_length=5,
        collision_energies: Union[List[float], float] = [27],
        charges: Union[List[int], int] = [2, 3],
    ) -> None:

        charges = [charges] if isinstance(charges, int) else charges
        collision_energies = (
            [collision_energies] if isinstance(charges, float) else collision_energies
        )

        fasta_file = Path(fasta_file)

        logging.info(
            (
                f"Processing file {fasta_file.name},"
                f" with enzyme={enzyme}, "
                f" missed_cleavages={missed_cleavages}"
                f" min_length={min_length}"
            )
        )

        sequences = []
        out_charges = []
        out_nces = []

        my_iter = self.yield_peptides(
            fasta_file=fasta_file,
            charges=charges,
            collision_energies=collision_energies,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            enzyme=enzyme,
        )

        for seq, charge, ce in my_iter:
            sequences.append(seq)
            out_charges.append(charge)
            out_nces.append(ce)

        super().__init__(
            sequences=sequences, collision_energies=out_nces, charges=out_charges
        )

    @staticmethod
    def yield_peptides(
        fasta_file, charges, collision_energies, missed_cleavages, min_length, enzyme
    ):
        unique_peptides_count = 0
        with open(fasta_file, mode="rt") as gzfile:
            for description, sequence in fasta.FASTA(gzfile):
                new_peptides = parser.cleave(
                    sequence,
                    rule=enzyme,
                    missed_cleavages=missed_cleavages,
                    min_length=min_length,
                )
                for charge in charges:
                    for ce in collision_energies:
                        for x in new_peptides:
                            if len(x) < 50:
                                unique_peptides_count += 1
                                yield x, charge, ce

        logging.info(f"Done, {unique_peptides_count} unique sequences")
