from __future__ import annotations

import warnings
import re
import logging
from pathlib import Path
from os import PathLike
from typing import Union, Optional, Generator, Iterator, NamedTuple

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
from pyteomics import mzml

import torch
from torch import Tensor

import elfragmentador.constants as CONSTANTS
from elfragmentador.spectra import Spectrum
from elfragmentador.model import PepTransformerModel
from elfragmentador.datasets.dataset import IterableDatasetBase, Predictor
from elfragmentador.named_batches import (
    PredictionResults,
    TrainBatch,
)
from elfragmentador.utils import _attempt_find_file, torch_batch_from_seq


class PinDataset(IterableDatasetBase):
    NUM_COLUMNS = 28
    REGEX_FILE_APPENDIX = re.compile("_\d+_\d+_\d+$")
    APPENDIX_CHARGE_REGEX = re.compile("(?<=_)\d+(?=_)")
    DOT_RE = re.compile("(?<=\.).*(?=\..*$)")
    TEMPLATE_STRING = "controllerType=0 controllerNumber=1 scan={SCAN_NUMBER}"
    DEFAULT_TENSOR = torch.tensor([0 for _ in range(CONSTANTS.NUM_FRAG_EMBEDINGS)])
    metric = "Xcorr"

    def __init__(
        self,
        in_path: PathLike,
        df: Optional[DataFrame] = None,
        nce_offset: float = 0,
    ):
        """Generate a Dataset from a percolator input file

        Args:
            in_path (PathLike): Input path to percolator input file
            df (DataFrame, optional): Pandas dataframe product of reading the file or a modification of it
            nce_offset (float, optional): [description]. Defaults to 0.

        """
        logging.info("Starting Percolator input dataset")
        self.in_path = Path(in_path)
        if df is None:
            # TODO fix so the last column remains unchanged, right now it keeps
            # only the first protein because the field is not quoted in comet
            # outputs
            df = pd.read_csv(
                in_path,
                sep="\t",
                index_col=False,
                usecols=list(range(PinDataset.NUM_COLUMNS)),
            )

            logging.info(f"Read DataFrame with columns {list(df)} and length {len(df)}")
        logging.info("Sorting input")

        if "PepLen" in list(df):
            column_order = ["PepLen", "Peptide", "CalcMass"]
        else:
            column_order = ["CalcMass", "Peptide"]

        df = df.sort_values(by=column_order).reset_index(drop=True).copy()
        self._initialize_on_df(df, nce_offset=nce_offset)

        # The appendix is in the form of _SpecNum_Charge_ID

    def _initialize_on_df(self, df, nce_offset):
        self.df = df

        self.mzml_readers = {}
        self.mzml_files = {}
        self.nce_offset = nce_offset

    def top_n_subset(self, n: int, ascending=False) -> PinDataset:
        """Generate another percolator dataset with a subset of the observations

        Args:
            n (int): Number of observations to return
            column (str): Column name to use to find the highest n
            ascending (bool, optional):
                Wether the observations should be the top or bottom.
                Defaults to False.

        Returns:
            PinDataset: Subsetted percolator dataset
        """
        df = self.df
        df = df.sort_values(self.metric, ascending=ascending).head(n)
        return self.__class__(df=df, in_path=self.in_path, nce_offset=self.nce_offset)

    def generate_elements(self) -> Generator[TrainBatch]:
        """Make a generator that goes though the percolator input

        The generator yields two named tuples with the inputs for a
        PepTransformerModel and the ground truth observed spectrum
        encoded as tensors.

        Yields:
            Generator[Tuple[ForwardBatch, PredictionResults]]: [description]
        """
        scan_id = None
        last_seq = None
        last_charge = None
        last_nce = None

        cached_reads = 0
        cached_batch_inputs = 0

        # TODO check if batching would improve inference speed
        for num_spec, (index, row) in enumerate(self.df.iterrows()):
            row_rawfile = re.sub(self.REGEX_FILE_APPENDIX, "", row.SpecId)
            row_appendix = self.REGEX_FILE_APPENDIX.search(row.SpecId)[0]

            curr_charge = int(self.APPENDIX_CHARGE_REGEX.search(row_appendix, 2)[0])
            peptide_sequence = self.DOT_RE.search(row.Peptide)[0]

            try:
                rawfile_path = self.mzml_files[row_rawfile]
            except KeyError as e:
                self.mzml_files[row_rawfile] = _attempt_find_file(
                    row_rawfile,
                    [
                        self.in_path.parent,
                        ".",
                        "../",
                        "../../",
                        self.in_path.parent / ".",
                        self.in_path.parent / "../",
                        self.in_path.parent / "../../",
                    ],
                )
                rawfile_path = self.mzml_files[row_rawfile]

            if self.mzml_readers.get(str(rawfile_path), None) is None:
                self.mzml_readers[str(rawfile_path)] = mzml.PreIndexedMzML(
                    str(rawfile_path)
                )

            old_scan_id = scan_id
            scan_id = self.TEMPLATE_STRING.format(SCAN_NUMBER=row.ScanNr)

            if old_scan_id != scan_id:
                # read_spectrum
                curr_scan = self.mzml_readers[str(rawfile_path)].get_by_id(scan_id)
                nce = float(
                    curr_scan["precursorList"]["precursor"][0]["activation"][
                        "collision energy"
                    ]
                )
                nce = self.calc_nce(nce)
                rt = float(curr_scan["scanList"]["scan"][0]["scan start time"])
            else:
                cached_reads += 1

            # convert spectrum to model "output"
            curr_spec_object = Spectrum(
                sequence=peptide_sequence,
                parent_mz=row.ExpMass,
                charge=curr_charge,
                mzs=curr_scan["m/z array"],
                intensities=curr_scan["intensity array"],
                nce=nce,
            )

            # predict spectrum
            if (
                last_seq != peptide_sequence
                or last_charge != curr_charge
                or last_nce != nce
            ):
                last_seq = peptide_sequence
                last_charge = curr_charge
                last_nce = nce

                input_batch = torch_batch_from_seq(
                    seq=peptide_sequence,
                    charge=curr_charge,
                    nce=nce,
                    enforce_length=False,
                    pad_zeros=False,
                )
            else:
                cached_batch_inputs += 1

            # Get ground truth spectrum
            try:
                gt_spec = torch.Tensor(curr_spec_object.encode_spectra())
            except AssertionError as e:
                if "No peaks were annotated in this spectrum" in str(e):
                    gt_spec = self.DEFAULT_TENSOR
                else:
                    raise AssertionError(e)

            expect_out = PredictionResults(irt=torch.tensor([rt]), spectra=gt_spec)
            squeezed_inputs = {
                k: v.squeeze(0) for k, v in input_batch._asdict().items()
            }

            out = TrainBatch(
                **squeezed_inputs, **expect_out._asdict(), weight=torch.ones(1)
            )

            yield out

        logging.info(
            (
                f"{num_spec + 1} Spectra Yielded,"
                f" {cached_batch_inputs} Cached inputs,"
                f" {cached_reads} Cached Spectrum reads"
            )
        )

    def greedify(self):
        logging.info(f"Making Greedy dataset of length {len(self)}")
        old_offset = self.nce_offset
        self.nce_offset = 0
        self.greedy_cache = [x for x in self]
        self.nce_offset = old_offset

    def greedy_iter(self):
        for input_batch in self.greedy_cache:
            input_batch = TrainBatch(
                seq=input_batch.seq,
                charge=input_batch.charge,
                mods=input_batch.mods,
                nce=input_batch.nce + self.nce_offset,
                spectra=input_batch.spectra,
                irt=input_batch.irt,
                weight=input_batch.weight,
            )
            yield input_batch

    def append_batches(self, batches: NamedTuple[Tensor], prefix=""):
        logging.info(f"Appending info to dataframe with prefix '{prefix}'")
        {
            logging.debug(f"Appending Batches: {prefix}{k}:{v.shape}")
            for k, v in batches._asdict().items()
        }
        for k, v in batches._asdict().items():
            k = prefix + k
            self.df.insert(loc=len(list(self.df)) - 2, column=k, value=float("nan"))
            self.df[k] = [x.numpy().flatten().tolist() for x in v]

    def save_data(self, prefix: PathLike):
        self.df.reset_index(drop=True).to_csv(prefix + ".csv", index=False)
        self.df.reset_index(drop=True).to_feather(prefix + ".feather")

    def __iter__(self) -> Iterator[TrainBatch]:
        if hasattr(self, "greedy_cache"):
            return self.greedy_iter()
        else:
            return self.generate_elements()

    def __len__(self):
        return len(self.df)


class MokapotPSMDataset(PinDataset):
    metric = "mokapot score"

    def __init__(
        self,
        in_path: PathLike,
        df: Optional[DataFrame] = None,
        nce_offset: float = 0,
        max_q: float = 0.01,
        max_pep: float = 0.01,
    ):
        """Generate a Dataset from a percolator input file

        Args:
            txt_file (PathLike): Path to a psms.txt file containing a rescored list of psms (output of mokapot)
            df (DataFrame, optional): Pandas dataframe product of reading the file or a modification of it
            nce_offset (float, optional): [description]. Defaults to 0.

        Details:
            Expects the columns:

            ```
             SpecId
             Label
             ScanNr
             ExpMass
             CalcMass
             Peptide
             mokapot score
             mokapot q-value
             mokapot PEP
             Proteins
            ```

        """
        logging.info("Starting Percolator output dataset")
        self.in_path = Path(in_path)
        if df is None:
            df = pd.read_csv(
                in_path,
                sep="\t",
                index_col=False,
            )

        logging.info(f"Read DataFrame with columns {list(df)} and length {len(df)}")

        df = df[df["mokapot q-value"] < max_q]
        df = df[df["mokapot PEP"] < max_pep]

        logging.info("Sorting input")
        self._initialize_on_df(df, nce_offset=nce_offset)


@torch.no_grad()
def append_preds(
    in_pin: Union[Path, str],
    out_pin: Union[Path, str],
    model: PepTransformerModel,
    predictor: Optional(Predictor) = None,
) -> pd.DataFrame:
    """Append cosine similarity to prediction to a percolator input

    Args:
        in_pin (Union[Path, str]): Input Percolator file location
        out_pin (Union[Path, str]): Output percolator file location
        model (PepTransformerModel): Transformer model to use

    Returns:
        pd.DataFrame: Pandas data frame with the appended column
    """

    # TODO check if this is needed
    warnings.filterwarnings(
        "ignore",
        ".*No peaks were annotated for spectra.*",
    )

    perc_inputs = PinDataset(in_path=in_pin)
    perc_inputs.optimize_nce(model, predictor=predictor)

    if predictor is None:
        predictor = Predictor()

    df = perc_inputs.df.copy()
    df.insert(loc=PinDataset.NUM_COLUMNS - 2, column="SpecCorrelation", value=0)
    df.insert(loc=PinDataset.NUM_COLUMNS - 2, column="DiffNormRT", value=100)

    outs = predictor.evaluate_dataset(model=model, dataset=perc_inputs)

    df["SpecAngle"] = 1 - outs.loss_angle
    df["NormRTSqError"] = outs.scaled_se_loss

    if out_pin is not None:
        df.to_csv(out_pin, index=False, sep="\t")

    return df
