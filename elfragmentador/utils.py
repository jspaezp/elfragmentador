import re
import random
from pathlib import Path
from typing import Union, Iterable
import logging

from pyteomics import mzml
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import elfragmentador
import elfragmentador.constants as CONSTANTS
from elfragmentador.spectra import Spectrum
from elfragmentador.model import PepTransformerModel
from elfragmentador.math_utils import norm

import torch
from torch.utils.data.dataset import TensorDataset
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

    # Read pin
    NUM_COLUMNS = 28
    compiled_model = model.to_torchscript()

    # The appendix is in the form of _SpecNum_Charge_ID
    regex_file_appendix = re.compile("_\d+_\d+_\d+$")
    appendix_charge_regex = re.compile("(?<=_)\d+(?=_)")
    dot_re = re.compile("(?<=\.).*(?=\..*$)")
    template_string = "controllerType=0 controllerNumber=1 scan={SCAN_NUMBER}"
    in_pin_path = Path(in_pin)

    df = pd.read_csv(
        in_pin,
        sep="\t",
        index_col=False,
        usecols=list(range(NUM_COLUMNS)),
    )
    # TODO fix so the last column remains unchanged, right now it keeps
    # only the first protein because the field is not quoted in comet
    # outputs

    logging.info(df)

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

    # This would allow skipping several reads on the file (which is fairly quick)
    # df = df.sort_values(by=['SpecId', 'ScanNr']).reset_index(drop=True).copy()
    # This would allow skipping several predictions... which right now are fairly slow
    df = df.sort_values(by=["Peptide", "CalcMass"]).reset_index(drop=True).copy()
    df.insert(loc=NUM_COLUMNS - 2, column="SpecCorrelation", value=0)
    df.insert(loc=NUM_COLUMNS - 2, column="DiffNormRT", value=100)

    mzml_readers = {}
    mzml_files = {}
    scan_id = None
    last_seq = None
    last_charge = None
    last_nce = None
    correlation_outs = []
    raw_rts = []
    pred_rts = []

    tqdm_postfix = {
        "cached_reads": 0,
        "cached_predictions": 0,
        "predictions": 0,
    }

    tqdm_iter = tqdm(df.iterrows(), total=len(df))

    # TODO check if batching would improve inference speed
    for index, row in tqdm_iter:
        row_rawfile = re.sub(regex_file_appendix, "", row.SpecId)
        row_appendix = regex_file_appendix.search(row.SpecId)[0]

        curr_charge = int(appendix_charge_regex.search(row_appendix, 2)[0])
        peptide_sequence = dot_re.search(row.Peptide)[0]

        try:
            rawfile_path = mzml_files[row_rawfile]
        except KeyError as e:
            rawfile_path = _attempt_find_file(
                row_rawfile, [in_pin_path.parent, ".", "../", "../../"]
            )

        if mzml_readers.get(str(rawfile_path), None) is None:
            mzml_readers[str(rawfile_path)] = mzml.PreIndexedMzML(str(rawfile_path))

        old_scan_id = scan_id
        scan_id = template_string.format(SCAN_NUMBER=row.ScanNr)

        if old_scan_id != scan_id:
            # read_spectrum
            curr_scan = mzml_readers[str(rawfile_path)].get_by_id(scan_id)
            nce = float(
                curr_scan["precursorList"]["precursor"][0]["activation"][
                    "collision energy"
                ]
            )
            rt = float(curr_scan["scanList"]["scan"][0]["scan start time"])
        else:

            tqdm_postfix["cached_reads"] += 1
            tqdm_iter.set_postfix(tqdm_postfix)

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

            input_batch = model.torch_batch_from_seq(
                seq=peptide_sequence,
                charge=curr_charge,
                nce=nce,
            )
            pred_irt, pred_spec = compiled_model(*input_batch)
            tqdm_postfix["predictions"] += 1
            tqdm_iter.set_postfix(tqdm_postfix)
        else:
            tqdm_postfix["cached_predictions"] += 1
            tqdm_iter.set_postfix(tqdm_postfix)

        # Get ground truth spectrum
        try:
            gt_spec = torch.Tensor(curr_spec_object.encode_spectra())

        except AssertionError as e:
            if "No peaks were annotated in this spectrum" in str(e):
                gt_spec = torch.zeros_like(pred_spec)

        # compare spectra
        distance = cosine_similarity(gt_spec, pred_spec, dim=-1)

        # append to results
        correlation_outs.append(float(distance))
        raw_rts.append(rt)
        pred_rts.append(float(pred_irt))

    df["SpecCorrelation"] = correlation_outs

    # TODO consider if i really need to make this an absolute value
    # making removing it would make it wose on percolater but
    # probably better on mokapot
    df["DiffNormRT"] = np.abs(norm(np.array(raw_rts))[0] - norm(np.array(pred_rts))[0])
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
