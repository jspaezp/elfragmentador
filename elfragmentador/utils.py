
import re
from pathlib import Path
from typing import Union

from pyteomics import mzml
import pandas as pd
from tqdm.auto import tqdm
from elfragmentador.spectra import Spectrum
from elfragmentador.model import PepTransformerModel

import torch
from torch.nn.functional import cosine_similarity

import warnings


def append_preds(in_pin: Union[Path, str], out_pin: Union[Path, str], model: PepTransformerModel) -> pd.DataFrame:
    """Append cosine similarity to prediction to a percolator input

    Args:
        in_pin (Union[Path, str]): Input Percolator file location
        out_pin (Union[Path, str]): Output percolator file location
        model (PepTransformerModel): Transformer model to use

    Returns:
        pd.DataFrame: Pandas data frame with the appended column
    """
    warnings.filterwarnings('ignore', '.*peaks were annotated for spectra.*', )
    # read pin
    NUM_COLUMNS = 28
    # The appendix is in the form of _SpecNum_Charge_ID
    regex_file_appendix = re.compile("_\d+_\d+_\d+$")
    appendix_charge_regex = re.compile("(?<=_)\d+(?=_)")
    dot_re = re.compile("(?<=\.).*(?=\..*$)")
    template_string = "controllerType=0 controllerNumber=1 scan={SCAN_NUMBER}"

    df = pd.read_csv(
        in_pin,
        sep = "\t",
        index_col = False,
        usecols=list(range(NUM_COLUMNS)),
    )
    # TODO fix so the last column remains unchanged, right now it keeps
    # only the first protein because the field is not quoted in comet
    # outputs

    print(df)
    df = df.sort_values(by=['SpecId', 'ScanNr']).reset_index(drop=True).copy()
    df.insert(loc = NUM_COLUMNS-2, column = "SpecCorrelation", value = 0)
    
    mzml_readers = {}
    scan_id = None
    outs = []
    
    for index, row in tqdm(df.iterrows(), total = len(df)):
        row_rawfile = re.sub(regex_file_appendix, "", row.SpecId)
        row_appendix = regex_file_appendix.search(row.SpecId)[0]
    
        curr_charge = int(appendix_charge_regex.search(row_appendix, 2)[0])
        peptide_sequence = dot_re.search(row.Peptide)[0]
    
        rawfile_path = Path(row_rawfile + ".mzML")
        assert rawfile_path.is_file(), f"{rawfile_path} does not exist"
        
        if mzml_readers.get(str(rawfile_path), None) is None:
            mzml_readers[str(rawfile_path)] = mzml.PreIndexedMzML(str(rawfile_path))
    
        old_scan_id = scan_id
        scan_id = template_string.format(SCAN_NUMBER = row.ScanNr)
        
        if old_scan_id != scan_id:
            # read_spectrum
            curr_scan = mzml_readers[str(rawfile_path)].get_by_id(scan_id)
            nce = curr_scan['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']
    
        # convert spectrum to model "output"
        curr_spec_object = Spectrum(
            sequence = peptide_sequence,
            parent_mz=row.ExpMass,
            charge = curr_charge,
            mzs = curr_scan["m/z array"],
            intensities = curr_scan["intensity array"],
            nce = nce)
    
        # predict spectrum
        with torch.no_grad():
            pred_irt, pred_spec = model.predict_from_seq(
                seq = peptide_sequence,
                charge = curr_charge,
                nce=nce,
            )
            pred_spec = torch.stack([pred_spec])

            # Get ground truth spectrum
            try:
                gt_spec = torch.stack([torch.Tensor(curr_spec_object.encode_spectra())])

            except AssertionError as e:
                if "No peaks were annotated in this spectrum" in str(e):
                    gt_spec = torch.zeros_like(pred_spec)
    

            # compare spectra
            distance = cosine_similarity(gt_spec, pred_spec)
    
        # append to results
        outs.append(float(distance))
    
    df["SpecCorrelation"] = outs
    df.to_csv(out_pin, index=False, sep="\t")
    return df
