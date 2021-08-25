from os import PathLike
import re
import random
from pathlib import Path
from typing import Union, Iterable, Tuple
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

    logging.error(f"File not found in any of: {tried_paths}")
    raise FileNotFoundError(tried_paths)


class PinDataset(IterableDataset):
    NUM_COLUMNS = 28
    REGEX_FILE_APPENDIX = re.compile("_\d+_\d+_\d+$")
    APPENDIX_CHARGE_REGEX = re.compile("(?<=_)\d+(?=_)")
    DOT_RE = re.compile("(?<=\.).*(?=\..*$)")
    TEMPLATE_STRING = "controllerType=0 controllerNumber=1 scan={SCAN_NUMBER}"

    def __init__(self, df: DataFrame, in_pin_path: PathLike, nce_offset: float=0):
        logging.info("Starting Percolator input dataset")
        logging.info(df)
        self.in_pin_path = Path(in_pin_path)

        logging.info("Sorting input")
        df = df.sort_values(by=["Peptide", "CalcMass"]).reset_index(drop=True).copy()
        self.df = df

        self.mzml_readers = {}
        self.mzml_files = {}
        self.nce_offset = nce_offset

        # The appendix is in the form of _SpecNum_Charge_ID
    
    def top_n_subset(self, n: int, column: str, ascending=False):
        df = self.df.sort_values(column, ascending = ascending).head(n)
        return PinDataset(df, self.in_pin_path, nce_offset=self.nce_offset)

    @staticmethod
    def from_path(in_pin, nce_offset=0):
        """
        Col names should be:
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
        """
        in_pin_path = Path(in_pin)
        # TODO fix so the last column remains unchanged, right now it keeps
        # only the first protein because the field is not quoted in comet
        # outputs
        df = pd.read_csv(
            in_pin_path,
            sep="\t",
            index_col=False,
            usecols=list(range(PinDataset.NUM_COLUMNS)),
        )
        return PinDataset(df, in_pin_path=in_pin_path, nce_offset=nce_offset)

    def generate_elements(self) -> Tuple[ForwardBatch, PredictionResults]:
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
                    row_rawfile, [self.in_pin_path.parent, ".", "../", "../../"]
                )

            if self.mzml_readers.get(str(rawfile_path), None) is None:
                self.mzml_readers[str(rawfile_path)] = mzml.PreIndexedMzML(str(rawfile_path))

            old_scan_id = scan_id
            scan_id = self.TEMPLATE_STRING.format(SCAN_NUMBER=row.ScanNr)

            if old_scan_id != scan_id:
                # read_spectrum
                curr_scan = self.mzml_readers[str(rawfile_path)].get_by_id(scan_id)
                nce = float(
                    curr_scan["precursorList"]["precursor"][0]["activation"][
                        "collision energy"
                    ]
                ) + self.nce_offset
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
                )
            else:
                cached_batch_inputs += 1

            # Get ground truth spectrum
            try:
                gt_spec = torch.Tensor(curr_spec_object.encode_spectra())
            except AssertionError as e:
                if "No peaks were annotated in this spectrum" in str(e):
                    gt_spec = torch.tensor([0 for _ in range(CONSTANTS.NUM_FRAG_EMBEDINGS)])
                else:
                    raise AssertionError(e)

            yield input_batch, PredictionResults(irt=torch.tensor(rt), spectra=gt_spec)

        logging.info((
            f"{num_spec} Spectra Yielded,"
            f" {cached_batch_inputs} Cached inputs,"
            f" {cached_reads} Cached Spectrum reads"
        ))

    def __iter__(self):
        return self.generate_elements()
        # Returns ForwardBatch, PredictionResults

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

    perc_inputs = PinDataset.from_path(in_pin)

    df = perc_inputs.df.copy()
    df.insert(loc=PinDataset.NUM_COLUMNS - 2, column="SpecCorrelation", value=0)
    df.insert(loc=PinDataset.NUM_COLUMNS - 2, column="DiffNormRT", value=100)

    correlation_outs = []
    gt_rts = []
    pred_rts = []
    last_fw_batch = ForwardBatch(
        torch.tensor(0).long(),
        torch.tensor(0).float(),
        torch.tensor(0).long(),
        torch.tensor(0).long(),
    )

    tqdm_iter = tqdm(perc_inputs, total=len(perc_inputs))
    tqdm_postfix = {
        "predictions" : 0,
        "cached_predictions" : 0,
    }

    # TODO check if batching would improve inference speed
    for forward_batch, gt_batch in tqdm_iter:
        # predict spectrum
        if not all([torch.equal(x,y) for x, y in zip(last_fw_batch, forward_batch)]):
            last_fw_batch = forward_batch
            pred_irt, pred_spec = compiled_model(*forward_batch)

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

    df["SpecCorrelation"] = correlation_outs

    # TODO consider if i really need to make this an absolute value
    # making removing it would make it wose on percolater but
    # probably better on mokapot
    df["DiffNormRT"] = np.abs(norm(np.array(gt_rts))[0] - norm(np.array(pred_rts))[0])
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
            seq=seq, charge=int(charge), nce=nce, as_spectrum=True
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
