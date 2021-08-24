import logging

from collections import namedtuple
import pandas as pd
from elfragmentador import annotate
import elfragmentador.constants as CONSTANTS
from pandas.core.frame import DataFrame
from torch import Tensor
from typing import Dict, List, Optional, Sequence, Union

SequencePair = namedtuple("SequencePair", "aas, mods")
SequencePair.__doc__ = """
Named Tuple that bundles aminoacid tensor encodings and its corresponding modifications.

Example for the following sequence `_AAIFVVAR_`:
> SequencePair(aas=[23, 1, 1, 8, 5, 19, 19, 1, 15, ..., 0], mods=[0, 0, 0, 0,..., 0, 0])
"""


def encode_mod_seq(seq: str) -> SequencePair:
    """
    Encodes a peptide sequence to a numeric vector

    Args:
        seq (str): Modified sequence to encode

    Raises:
        ValueError: Raises this error if the provided sequence
            is longer than the maximum allowed for the model.

    Examples:
        >>> samp_seq = "_AAIFVVAR_"
        >>> print(CONSTANTS.MAX_TENSOR_SEQUENCE)
        32
        >>> out = encode_mod_seq(samp_seq)
        >>> out
        SequencePair(aas=[23, 1, 1, 8, 5, 19, 19, 1, 15, ..., 0], mods=[0, 0, 0, 0,..., 0, 0])
        >>> len(out)
        2
        >>> [len(x) for x in out]
        [32, 32]
    """
    seq_out = [0] * CONSTANTS.MAX_TENSOR_SEQUENCE
    mod_out = [0] * CONSTANTS.MAX_TENSOR_SEQUENCE

    try:
        split_seq = list(annotate.peptide_parser(seq, solve_aliases=True))
        seq_out_i = [CONSTANTS.ALPHABET[x[:1]] for x in split_seq]
        mod_out_i = [
            CONSTANTS.MOD_PEPTIDE_ALIASES[x] if len(x) > 1 else 0 for x in split_seq
        ]
        mod_out_i = [CONSTANTS.MOD_INDICES.get(x, 0) for x in mod_out_i]
        if len(seq_out_i) > len(seq_out):
            logging.warning(
                f"Length of the encoded sequence"
                f" is more than the one allowed {CONSTANTS.MAX_SEQUENCE}."
                f" Sequence={seq}, the remainder will be clipped"
            )

        seq_out[: len(seq_out_i)] = seq_out_i
        mod_out[: len(mod_out_i)] = mod_out_i
    except ValueError as e:
        logging.error(seq)
        logging.error(e)
        raise ValueError(
            f"Sequence provided is longer than the supported length of {CONSTANTS.MAX_SEQUENCE}"
        )

    return SequencePair(seq_out, mod_out)


def clip_explicit_terminus(seq: Union[str, List]):
    """Remove explicit terminus

    Args:
        seq (Union[str, List]): Sequence to be stripped form eplicit termini

    Returns:
        Sequence (Union[str, List]): Same as sequence input but removing explicit n and c termini

    Examples:
        >>> clip_explicit_terminus("PEPTIDEPINK")
        'PEPTIDEPINK'
        >>> clip_explicit_terminus("nPEPTIDEPINKc")
        'PEPTIDEPINK'
        >>> clip_explicit_terminus("n[ACETYL]PEPTIDEPINKc")
        'n[ACETYL]PEPTIDEPINK'
    """

    if seq[0] == "n" and not seq[1].startswith("["):
        seq = seq[1:]

    if seq[-1] == "c":
        seq = seq[:-1]

    return seq


def decode_mod_seq(
    seq_encoding: List[int],
    mod_encoding: Optional[List[int]] = None,
    clip_explicit_term=True,
) -> str:
    """Decode a pair of encoded sequences to a string representation

    Args:
        seq_encoding (List[int]):
            List of integers encoding a peptide sequence
        mod_encoding (Optional[List[int]], optional):
            List of integers representing the modifications on the sequence. Defaults to None.
        clip_explicit_term (bool, optional):
            Wether the explicit n and c terminus should be included. Defaults to True.

    Returns:
        str: String sequence with the peptide

    Examples:
        >>> decode_mod_seq(seq_encoding=[1,1,1], mod_encoding=[0,1,0])
        'AA[CARBAMIDOMETHYL]A'
    """
    out = []

    if mod_encoding is None:
        mod_encoding = [0] * len(seq_encoding)

    for i, s in enumerate(seq_encoding):
        if s == 0:
            break

        out.append(CONSTANTS.ALPHABET_S[s])
        if mod_encoding[i] != 0:
            out.append(f"[{CONSTANTS.MOD_INDICES_S[mod_encoding[i]]}]")

    if clip_explicit_term:
        out = clip_explicit_terminus(out)
    return "".join(out)


def encode_fragments(
    annotated_peaks: Optional[Union[Dict[str, int], Dict[str, float]]]
) -> Union[List[float], List[int]]:
    """
    Gets either the labels or an sequence that encodes a spectra
    # TODO split this into different functions ...

    Examples:
        >>> encode_fragments({'z1y2': 100, 'z2y2': 52})
        [0, 0, 0, 100, ..., 0, 52, ...]
    """

    # TODO implement neutral losses ...  if needed
    encoding = [annotated_peaks.get(key, 0) for key in CONSTANTS.FRAG_EMBEDING_LABELS]

    return encoding


def decode_fragment_tensor(
    sequence: str,
    tensor: Union[List[int], Tensor],
) -> DataFrame:
    """
    Returns a data frame with annotations from sequence
    and a tensor encoding a spectra

    Examples:
        >>> import torch
        >>> foo = decode_fragment_tensor("AAACK", torch.arange(0, (CONSTANTS.NUM_FRAG_EMBEDINGS)))
        >>> foo.head()
        Fragment        Mass  Intensity
        0     z1b1   72.044390        0.0
        1     z1y1  147.112804        1.0
        2     z1b2  143.081504        2.0
        3     z1y2  307.143453        3.0
        4     z1b3  214.118618        4.0
        >>> # import matplotlib.pyplot as plt
        >>> # plt.vlines(foo['Mass'], 0, foo['Intensity'])
        >>> # plt.show()
    """
    key_list = CONSTANTS.FRAG_EMBEDING_LABELS
    fragment_ions = annotate.get_peptide_ions(sequence)
    masses = [fragment_ions.get(k, 0) for k in key_list]
    intensities = [float(x) for x in tensor]

    assert len(intensities) == len(masses), logging.error(
        f"Int {len(intensities)}: \n{intensities}\n\nmasses {len(masses)}: \n{masses}"
    )

    out_dict = {"Fragment": key_list, "Mass": masses, "Intensity": intensities}
    out_df = pd.DataFrame(out_dict)
    out_df = out_df[out_df["Mass"] != 0].copy()

    return out_df
