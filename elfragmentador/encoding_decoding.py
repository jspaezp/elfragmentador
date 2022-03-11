import logging
from collections import namedtuple
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

import elfragmentador.constants as CONSTANTS
from elfragmentador import annotate

SequencePair = namedtuple("SequencePair", "aas, mods")
SequencePair.__doc__ = """
Named Tuple that bundles aminoacid tensor encodings and its corresponding modifications.

Example for the following sequence `_AAIFVVAR_`:
> SequencePair(aas=[23, 1, 1, 8, 5, 19, 19, 1, 15, ..., 0], mods=[0, 0, 0, 0,..., 0, 0])
"""


def encode_mod_seq(seq: str, enforce_length=True, pad_zeros=True) -> SequencePair:
    """
    Encodes a peptide sequence to a numeric vector.

    Args:
        seq (str): Modified sequence to encode
        enforce_length (bool):
            Wether to assure if the length of the tensors returned
            match the supported sequence outputs.

    Raises:
        ValueError: Raises this error if the provided sequence
            is longer than the maximum allowed for the model.

    Examples:
        >>> samp_seq = "_AAIFVVAR_"
        >>> print(CONSTANTS.MAX_TENSOR_SEQUENCE)
        32
        >>> out = encode_mod_seq(samp_seq)
        >>> out
        SequencePair(aas=[23, 1, 1, 8, 5, 19, 19, 1, 15, ..., 0],
           mods=[0, 0, 0, 0,..., 0, 0])
        >>> len(out)
        2
        >>> [len(x) for x in out]
        [32, 32]
        >>> [len(x) for x in encode_mod_seq("A" * 200)]
        [32, 32]
        >>> [len(x) for x in encode_mod_seq("A" * 200, enforce_length=False)]
        [202, 202]
        >>> [len(x) for x in encode_mod_seq("A" * 5, pad_zeros=False)]
        [7, 7]
    """
    seq = annotate.canonicalize_seq(seq)
    split_seq = list(annotate.peptide_parser(seq, solve_aliases=True))

    seq_out = [0] * (CONSTANTS.MAX_TENSOR_SEQUENCE if pad_zeros else len(split_seq))
    mod_out = seq_out.copy()

    seq_out_i = [CONSTANTS.ALPHABET[x[:1]] for x in split_seq]
    mod_out_i = [
        CONSTANTS.MOD_PEPTIDE_ALIASES[x] if len(x) > 1 else 0 for x in split_seq
    ]
    mod_out_i = [CONSTANTS.MOD_INDICES.get(x, 0) for x in mod_out_i]

    try:
        if (len(seq_out_i) > len(seq_out)) and enforce_length:
            logging.warning(
                f"Length of the encoded sequence"
                f" is more than the one allowed {CONSTANTS.MAX_SEQUENCE}."
                f" Sequence={seq}, the remainder will be clipped"
            )
            seq_out_i = seq_out_i[: CONSTANTS.MAX_TENSOR_SEQUENCE]
            mod_out_i = mod_out_i[: CONSTANTS.MAX_TENSOR_SEQUENCE]

        seq_out[: len(seq_out_i)] = seq_out_i
        mod_out[: len(mod_out_i)] = mod_out_i
    except ValueError as e:
        logging.error(f"Handling error '{e}' on sequence={seq}")
        raise ValueError(e)

    return SequencePair(seq_out, mod_out)


def clip_explicit_terminus(seq: Union[str, List]):
    """
    Remove explicit terminus.

    Args:
        seq (Union[str, List]): Sequence to be stripped form eplicit termini

    Returns:
        Sequence (Union[str, List]):
            Same as sequence input but removing explicit n and c termini

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
    """
    Decode a pair of encoded sequences to a string representation.

    Args:
        seq_encoding (List[int]):
            List of integers encoding a peptide sequence
        mod_encoding (Optional[List[int]], optional):
            List of integers representing the modifications on the sequence.
            Defaults to None.
        clip_explicit_term (bool, optional):
            Wether the explicit n and c terminus should be included. Defaults to True.

    Returns:
        str: String sequence with the peptide

    Examples:
        >>> decode_mod_seq(seq_encoding=[1,1,1], mod_encoding=[0,1,0])
        'AA[CARBAMIDOMETHYL]A'
    """
    out = []

    if mod_encoding is None or len(mod_encoding) == 0:
        mod_encoding = [0] * len(seq_encoding)

    for i, s in enumerate(seq_encoding):
        # This solves the issue where a tensor is passed instead
        # of a list of ints
        s = int(s)
        if s == 0:
            break

        out.append(CONSTANTS.ALPHABET_S[s])
        if mod_encoding[i] != 0:
            out.append(f"[{CONSTANTS.MOD_INDICES_S[int(mod_encoding[i])]}]")

    if clip_explicit_term:
        out = clip_explicit_terminus(out)
    return "".join(out)


def encode_fragments(
    annotated_peaks: Optional[Union[Dict[str, int], Dict[str, float]]]
) -> Union[List[float], List[int]]:
    """
    Gets either the labels or an sequence that encodes a spectra.

    # TODO split this into different functions ...

    Examples:
        >>> encode_fragments({'z1y2': 100, 'z2y2': 52})
        [0, 0, 0, 100, ..., 0, 52, ...]
    """

    # TODO implement neutral losses ...  if needed
    encoding = [annotated_peaks.get(key, 0) for key in CONSTANTS.FRAG_EMBEDING_LABELS]

    return encoding


@torch.no_grad()
def decode_fragment_tensor(
    sequence: str,
    tensor: Union[List[int], Tensor],
) -> Dict[str, Union[List[str], np.float32]]:
    """
    Returns a data frame with annotations from sequence and a tensor encoding a
    spectra.

    Examples:
        >>> import torch
        >>> foo = decode_fragment_tensor(
        ... "AAACK",
        ... torch.arange(0, (CONSTANTS.NUM_FRAG_EMBEDINGS)))
        >>> {k:v[:5] for k,v in foo.items()}
        {'Fragment': ['z1b1', 'z1y1', 'z1b2', 'z1y2', 'z1b3'], \
'Mass': array([ 72.04439047, 147.11280417, 143.08150447, 307.14345289, \
214.11861847]), 'Intensity': array([0., 1., 2., 3., 4.])}
    """
    key_list = CONSTANTS.FRAG_EMBEDING_LABELS
    fragment_ions = annotate.get_peptide_ions(sequence)
    masses = [fragment_ions.get(k, 0) for k in key_list]
    intensities = (
        [float(x) for x in tensor]
        if isinstance(tensor, list)
        else tensor.float().cpu().numpy()
    )

    assert len(intensities) == len(masses), logging.error(
        f"Int {len(intensities)}: \n{intensities}\n\nmasses {len(masses)}: \n{masses}"
    )

    key_list = [k for k, i in zip(key_list, masses) if i > 0]
    intensities = np.float64([k for k, i in zip(intensities, masses) if i > 0])
    masses = np.float64([k for k, i in zip(masses, masses) if i > 0])

    out_dict = {"Fragment": key_list, "Mass": masses, "Intensity": intensities}

    return out_dict
