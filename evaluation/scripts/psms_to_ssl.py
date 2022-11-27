import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger as lg_logger
from tqdm.auto import tqdm


def convert_to_ssl(input_file, input_peptides, output_file):
    lg_logger.info(f"Reading File {input_peptides}")
    pep_df = pd.read_csv(input_peptides, sep="\t")
    pep_df = pep_df[pep_df["mokapot q-value"] < 0.01]
    peps = set(pep_df["Peptide"])

    lg_logger.info(f"Reading File {input_file}")
    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype={
            "SpecId": str,
            "Label": bool,
            "ScanNr": np.int32,
            "ExpMass": np.float16,
            "CalcMass": np.float16,
            "Peptide": str,
            "mokapot score": np.float16,
            "mokapot q-value": np.float32,
            "mokapot PEP": np.float32,
            "Proteins": str,
        },
    )

    lg_logger.info(f"Processing File {df.shape}")
    df = df[df["mokapot q-value"] < 0.01]
    lg_logger.info(f"Processing File {df.shape} filtered pval")
    df = df[[pep in peps for pep in tqdm(df["Peptide"])]]
    lg_logger.info(f"Processing File {df.shape} filtered peptides pval")
    df2 = []
    for i, x in tqdm(df.groupby("Peptide")):
        tmp = x.sort_values(by="mokapot score", ascending=False).head(20)
        df2.append(tmp)

    df = pd.concat(df2)

    lg_logger.info(f"Processing File {df.shape} filtered top")
    out_df = pd.DataFrame([_parse_line(x) for _, x in tqdm(df.iterrows())])

    lg_logger.info(f"Writting output: {output_file}")
    out_df.to_csv(output_file, sep="\t", index=False, header=True)

    lg_logger.info("Done")


SPEC_ID_REGEX = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)$")


def _parse_line(line):
    outs = SPEC_ID_REGEX.match(line.SpecId).groups()
    file_name, spec_number, charge, _ = outs
    file_name = Path(file_name).stem + ".mzML"

    first_period = 0
    last_period = 0
    for i, x in enumerate(line.Peptide):
        if x == ".":
            if first_period == 0:
                first_period = i
            else:
                last_period = i

    sequence = line.Peptide[first_period + 1 : last_period]

    line_out = {
        "file": file_name,
        "scan": spec_number,
        "charge": charge,
        "sequence": convert_sequence(sequence),
        "score-type": "PERCOLATOR QVALUE",
        "score": line["mokapot q-value"],
    }
    return line_out


N_TERM_MOD_REGEX = re.compile(r"n\[([0-9.]+)\](.)((\[([0-9.]+)\])?(.*))")


def convert_sequence(sequence):
    """
    >>> convert_sequence('n[229.1629]K[229.1629]AAAAAAALQA')
    'K[458.3258]AAAAAAALQA'
    >>> convert_sequence('K[229.1629]AAAAAAALQA')
    'K[229.1629]AAAAAAALQA'
    >>> convert_sequence('n[229.1629]KAAAAAAALQA')
    'K[229.1629]AAAAAAALQA'
    >>> convert_sequence('KAAAAAAALQA')
    'KAAAAAAALQA'
    """

    match = N_TERM_MOD_REGEX.match(sequence)
    if not match:
        return sequence

    new_mass = float(match.group(1)) + float(match.group(5) or 0)
    out = f"{match.group(2)}[{new_mass:.04f}]{match.group(6)}"
    return out


def test_convert_sequence():
    inp = "n[229.1629]K[229.1629]AAAAAAALQA"
    out = "K[458.3258]AAAAAAALQA"
    assert convert_sequence(inp) == out
    assert (
        convert_sequence("n[229.1629]K[229.1629]AAAAAAALQA") == "K[458.3258]AAAAAAALQA"
    )
    assert convert_sequence("K[229.1629]AAAAAAALQA") == "K[229.1629]AAAAAAALQA"
    assert convert_sequence("n[229.1629]KAAAAAAALQA") == "K[229.1629]AAAAAAALQA"
    assert convert_sequence("KAAAAAAALQA") == "KAAAAAAALQA"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a psms file to ssl format")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file, should be a .mokapot.psms.txt file",
    )
    parser.add_argument(
        "--input_peptides",
        type=str,
        required=True,
        help="Input file, should be a .mokapot.peptides.txt file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file in ssl format",
    )
    args = parser.parse_args()

    convert_to_ssl(args.input_file, args.input_peptides, args.output_file)
