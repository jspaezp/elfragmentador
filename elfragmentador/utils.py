from __future__ import annotations

from os import PathLike
import re
import random
from pathlib import Path
from typing import Union, Iterable, Tuple, Optional, Generator
import logging
from pandas.core.frame import DataFrame

from pyteomics import mzml
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import elfragmentador
import elfragmentador.constants as CONSTANTS
from elfragmentador.spectra import Spectrum
from elfragmentador.model import ForwardBatch, PepTransformerModel, PredictionResults
from elfragmentador.math_utils import norm

import torch
from torch.utils.data.dataset import IterableDataset, TensorDataset
from torch.nn.functional import cosine_similarity

import warnings


# TODO split addition of metadata and actual predictions to separate functions to


def _attempt_find_file(row_rawfile, possible_paths):
    tried_paths = []
    for pp in possible_paths:
        rawfile_path = Path(pp) / (row_rawfile + ".mzML")
        tried_paths.append(rawfile_path)

        if rawfile_path.is_file():
            return rawfile_path
        else:
            logging.debug(f"{rawfile_path}, not found")

    logging.error(f"File not found in any of: {[str(x) for x in tried_paths]}")
    raise FileNotFoundError(tried_paths)


class PinDataset(IterableDataset):
    NUM_COLUMNS = 28
    REGEX_FILE_APPENDIX = re.compile("_\d+_\d+_\d+$")
    APPENDIX_CHARGE_REGEX = re.compile("(?<=_)\d+(?=_)")
    DOT_RE = re.compile("(?<=\.).*(?=\..*$)")
    TEMPLATE_STRING = "controllerType=0 controllerNumber=1 scan={SCAN_NUMBER}"
    DEFAULT_TENSOR = torch.tensor([0 for _ in range(CONSTANTS.NUM_FRAG_EMBEDINGS)])

    def __init__(
        self,
        in_pin_path: PathLike,
        df: Optional[DataFrame] = None,
        nce_offset: float = 0,
    ):
        """Generate a Dataset from a percolator input file

        Args:
            in_pin_path (PathLike): Input path to percolator input file
            df (DataFrame, optional): Pandas dataframe product of reading the file or a modification of it
            nce_offset (float, optional): [description]. Defaults to 0.

        """
        logging.info("Starting Percolator input dataset")
        self.in_pin_path = Path(in_pin_path)
        if df is None:
            # TODO fix so the last column remains unchanged, right now it keeps
            # only the first protein because the field is not quoted in comet
            # outputs
            df = pd.read_csv(
                in_pin_path,
                sep="\t",
                index_col=False,
                usecols=list(range(PinDataset.NUM_COLUMNS)),
            )

        logging.info(f"Read DataFrame with columns {list(df)} and length {len(df)}")
        logging.info("Sorting input")
        df = df.sort_values(by=["Peptide", "CalcMass"]).reset_index(drop=True).copy()
        self.df = df

        self.mzml_readers = {}
        self.mzml_files = {}
        self.nce_offset = nce_offset

        # The appendix is in the form of _SpecNum_Charge_ID

    def top_n_subset(self, n: int, column: str, ascending=False) -> PinDataset:
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
        df = df[df["PepLen"] < CONSTANTS.MAX_SEQUENCE]
        df = df.sort_values(column, ascending=ascending).head(n)
        return PinDataset(
            df=df, in_pin_path=self.in_pin_path, nce_offset=self.nce_offset
        )

    def generate_elements(self) -> Generator[Tuple[ForwardBatch, PredictionResults]]:
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
                rawfile_path = _attempt_find_file(
                    row_rawfile,
                    [
                        self.in_pin_path.parent,
                        ".",
                        "../",
                        "../../",
                        self.in_pin_path.parent / ".",
                        self.in_pin_path.parent / "../",
                        self.in_pin_path.parent / "../../",
                    ],
                )

            if self.mzml_readers.get(str(rawfile_path), None) is None:
                self.mzml_readers[str(rawfile_path)] = mzml.PreIndexedMzML(
                    str(rawfile_path)
                )

            old_scan_id = scan_id
            scan_id = self.TEMPLATE_STRING.format(SCAN_NUMBER=row.ScanNr)

            if old_scan_id != scan_id:
                # read_spectrum
                curr_scan = self.mzml_readers[str(rawfile_path)].get_by_id(scan_id)
                nce = (
                    float(
                        curr_scan["precursorList"]["precursor"][0]["activation"][
                            "collision energy"
                        ]
                    )
                    + self.nce_offset
                )
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

                input_batch = PepTransformerModel.torch_batch_from_seq(
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

            yield input_batch, PredictionResults(irt=torch.tensor(rt), spectra=gt_spec)

        logging.info(
            (
                f"{num_spec + 1} Spectra Yielded,"
                f" {cached_batch_inputs} Cached inputs,"
                f" {cached_reads} Cached Spectrum reads"
            )
        )

    def optimize_nce(self, scripted_model, offsets=range(-10, 10, 6), n=500):
        offsets = [0] + list(offsets)
        logging.info(f"Finding best nce offset from {offsets}")
        best = 0
        best_score = 0

        tmp_ds = self.top_n_subset(n=n, column="Xcorr")
        tmp_ds.greedify()
        for i, offset in enumerate(offsets):
            tmp_ds.nce_offset = offset
            out_spec, out_rt = tmp_ds.compare_predicted(scripted_model, verbose=False)
            score = np.mean(out_spec)
            logging.info(f"NCE Offset={offset}, score={score}")
            if score > best_score:
                if i > 0:
                    logging.info(
                        (
                            f"Updating best offset (from {best} to {offset}) "
                            f"because {best_score} < {score}"
                        )
                    )
                best = offset
                best_score = score

        logging.info(f"Best nce offset was {best}, score = {best_score}")
        self.nce_offset = best
        return best

    @torch.no_grad()
    def compare_predicted(self, scripted_model, verbose=True):
        correlation_outs = []
        gt_rts = []
        pred_rts = []
        last_fw_batch = ForwardBatch(
            torch.tensor(0).long(),
            torch.tensor(0).float(),
            torch.tensor(0).long(),
            torch.tensor(0).long(),
        )

        tqdm_iter = tqdm(self, total=len(self), disable=not verbose)
        tqdm_postfix = {
            "predictions": 0,
            "cached_predictions": 0,
        }

        # TODO check if batching would improve inference speed
        for forward_batch, gt_batch in tqdm_iter:
            # predict spectrum
            if not all(
                [torch.equal(x, y) for x, y in zip(last_fw_batch, forward_batch)]
            ):
                last_fw_batch = forward_batch
                pred_irt, pred_spec = scripted_model(*forward_batch)

                tqdm_postfix["predictions"] += 1
                tqdm_iter.set_postfix(tqdm_postfix)
            else:
                tqdm_postfix["cached_predictions"] += 1
                tqdm_iter.set_postfix(tqdm_postfix)

            # compare spectra
            distance = cosine_similarity(gt_batch.spectra, pred_spec, dim=-1)

            # append to results
            correlation_outs.append(float(distance))
            gt_rts.append(float(gt_batch.irt))
            pred_rts.append(float(pred_irt))

        # TODO consider if i really need to make this an absolute value
        # making removing it would make it wose on percolater but
        # probably better on mokapot
        rt_diff_outs = np.abs(norm(np.array(gt_rts))[0] - norm(np.array(pred_rts))[0])
        correlation_outs = np.array(correlation_outs)

        return correlation_outs, rt_diff_outs

    def greedify(self):
        logging.info(f"Making Greedy dataset of length {len(self)}")
        old_offset = self.nce_offset
        self.nce_offset = 0
        self.greedy_cache = [x for x in self]
        self.nce_offset = old_offset

    def greedy_iter(self):
        for input_batch, gt_batch in self.greedy_cache:
            input_batch = ForwardBatch(
                src=input_batch.src,
                charge=input_batch.charge,
                mods=input_batch.mods,
                nce=input_batch.nce + self.nce_offset,
            )
            yield input_batch, gt_batch

    def __iter__(self):
        if hasattr(self, "greedy_cache"):
            return self.greedy_iter()
        else:
            return self.generate_elements()

    def __len__(self):
        return len(self.df)


@torch.no_grad()
def append_preds(
    in_pin: Union[Path, str], out_pin: Union[Path, str], model: PepTransformerModel
) -> pd.DataFrame:
    """Append cosine similarity to prediction to a percolator input

    Args:
        in_pin (Union[Path, str]): Input Percolator file location
        out_pin (Union[Path, str]): Output percolator file location
        model (PepTransformerModel): Transformer model to use

    Returns:
        pd.DataFrame: Pandas data frame with the appended column
    """

    warnings.filterwarnings(
        "ignore",
        ".*peaks were annotated for spectra.*",
    )

    compiled_model = model.to_torchscript()
    compiled_model.eval()

    perc_inputs = PinDataset(in_pin_path=in_pin)
    perc_inputs.optimize_nce(model)

    df = perc_inputs.df.copy()
    df.insert(loc=PinDataset.NUM_COLUMNS - 2, column="SpecCorrelation", value=0)
    df.insert(loc=PinDataset.NUM_COLUMNS - 2, column="DiffNormRT", value=100)

    correlation_outs, rt_diff_outs = perc_inputs.compare_predicted(compiled_model)

    df["SpecCorrelation"] = correlation_outs
    df["DiffNormRT"] = rt_diff_outs

    if out_pin is not None:
        df.to_csv(out_pin, index=False, sep="\t")

    return df


@torch.no_grad()
def predict_df(
    df: pd.DataFrame, impute_collision_energy=False, model: PepTransformerModel = None
) -> str:
    """
    Predicts the spectra from a precursor list as defined by Skyline

    Args:
        df (pd.DataFrame):
          A data frame containing minimum 3 columns
          modified_sequence ('Modified Sequence'),
          collision_energy ('CE'),
          precursor_charge ('Precursor Charge')
        impute_collision_energy (Union[bool, float]):
          Either False or a collision energy to use
          predicting the spectra
    """
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

    if names[1] not in list(df):
        if impute_collision_energy:
            df[names[1]] = impute_collision_energy
        else:
            raise ValueError(
                f"Didn't find a collision enery column with name {names[1]},"
                " please provide one or a value for `impute_collision_energy`"
            )

    if model is None:
        model = PepTransformerModel.load_from_checkpoint(
            elfragmentador.DEFAULT_CHECKPOINT
        )

    my_iter = tqdm(zip(df[names[0]], df[names[1]], df[names[2]]), total=len(df))
    out = []

    for seq, nce, charge in my_iter:
        pred_spec = model.predict_from_seq(
            seq=seq, charge=int(charge), nce=nce, as_spectrum=True, enforce_length=False
        )
        out.append(pred_spec.to_sptxt())

    return "\n".join(out)


def get_random_peptide():
    AAS = [x for x in CONSTANTS.ALPHABET if x.isupper()]
    len_pep = random.randint(5, CONSTANTS.MAX_SEQUENCE)
    out_pep = ""

    for _ in range(len_pep):
        out_pep += "".join(random.sample(AAS, 1))

    return out_pep


def _concat_batches(batches):
    out = []
    for i, _ in enumerate(batches[0]):
        out.append(torch.cat([b[i] for b in batches]))

    return tuple(out)


def prepare_fake_tensor_dataset(num=50):
    peps = [
        {
            "nce": 20 + (10 * random.random()),
            "charge": random.randint(1, 5),
            "seq": get_random_peptide(),
        }
        for _ in range(num)
    ]

    tensors = [PepTransformerModel.torch_batch_from_seq(**pep) for pep in peps]
    tensors = TensorDataset(*_concat_batches(batches=tensors))

    return tensors
