"""
Greatly inspired/copied from:
https://github.com/kusterlab/prosit/blob/master/prosit/constants.py

And released under an Apache 2.0 license
"""

import numpy
import collections
from transprosit import constants
from collections import OrderedDict
from numpy import bool_, float64, ndarray
from typing import Iterator


def peptide_parser(p: str) -> Iterator[str]:
    """
    Parses maxquant formatted peptide strings

    Examples
    ========
    >>> list(peptide_parser("AAACC"))
    ['A', 'A', 'A', 'C', 'C']
    >>> list(peptide_parser("AAAM(ox)CC"))
    ['A', 'A', 'A', 'M(ox)', 'C', 'C']
    >>> list(peptide_parser("AAAM[+16]CC"))
    ['A', 'A', 'A', 'M[+16]', 'C', 'C']
    """

    ANNOTATIONS = "[](){}"

    if p[0] in ANNOTATIONS:
        raise ValueError(f"sequence starts with '{p[0]}'")
    n = len(p)
    i = 0
    while i < n:
        if p[i] == "_":
            i += 1
            continue
        elif i + 1 < n and p[i + 1] in ANNOTATIONS:
            p_ = p[i + 2 :]
            annots = [x for x in ANNOTATIONS if x in p_]
            nexts = []
            for an in annots:
                nexts.append(p_.index(an))
            j = min(nexts)
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1


def get_precursor_mz(peptide: str, charge: int):
    """
    Calcultes the theoretical mass of a precursor peptide
    (assumes positive mode)
    """
    raise NotImplementedError


def get_forward_backward(peptide: str):
    """
    Calculates masses forward and backwards from aminoacid
    sequences

    Examples
    ========
    >>> get_forward_backward("AMC")
    (array([ 71.037114  , 202.077599  , 362.10824772]), array([160.03064872, 291.07113372, 362.10824772]))
    >>> get_forward_backward("AM[147]C")
    (array([ 71.037114  , 218.072509  , 378.10315772]), array([160.03064872, 307.06604372, 378.10315772]))
    >>> get_forward_backward("AMC")
    (array([ 71.037114  , 202.077599  , 362.10824772]), array([160.03064872, 291.07113372, 362.10824772]))
    """
    amino_acids = peptide_parser(peptide)
    masses = [constants.MOD_AA_MASSES[a] for a in amino_acids]
    forward = numpy.cumsum(masses)
    backward = numpy.cumsum(list(reversed(masses)))
    return forward, backward


def get_mz(sum_: float64, ion_offset: float, charge: int) -> float64:
    return (sum_ + ion_offset + charge * constants.PROTON) / charge


def get_mzs(cumsum, ion_type, z):
    return [get_mz(s, constants.ION_OFFSET[ion_type], z) for s in cumsum[:-1]]


def get_annotation(
    forward: ndarray, backward: ndarray, charge: int, ion_types: str
) -> OrderedDict:
    """
    Calculates the ion annotations based on the forward
    and backward cumulative masses

    Example
    =======
    >>> fw, bw  = get_forward_backward("AMC")
    >>> fw
    array([ 71.037114  , 202.077599  , 362.10824772])
    >>> bw
    array([160.03064872, 291.07113372, 362.10824772])
    >>> get_annotation(fw, bw, 3, "y")
    OrderedDict([('y1', 60.354347608199994), ..., ('y2-NH3', 98.35899290653332)])
    """
    tmp = "{}{}"
    tmp_nl = "{}{}-{}"
    all_ = {}
    for ion_type in ion_types:
        if ion_type in constants.FORWARD:
            cummass = forward
        elif ion_type in constants.BACKWARD:
            cummass = backward
        else:
            raise ValueError("unkown ion_type: {}".format(ion_type))
        masses = get_mzs(cummass, ion_type, charge)
        d = {tmp.format(ion_type, i + 1): m for i, m in enumerate(masses)}
        all_.update(d)
        for nl, offset in constants.NEUTRAL_LOSS.items():
            nl_masses = get_mzs(cummass - offset, ion_type, charge)
            d = {tmp_nl.format(ion_type, i + 1, nl): m for i, m in enumerate(nl_masses)}
            all_.update(d)
    return collections.OrderedDict(sorted(all_.items(), key=lambda t: t[0]))


def get_peptide_ions(aa_seq):
    out = _get_peptide_ions(
        aa_seq,
        charges=range(1, constants.MAX_FRAG_CHARGE + 1),
        ion_types=constants.ION_TYPES,
    )
    return out


def _get_peptide_ions(aa_seq, charges=range(1, 5), ion_types="yb"):
    """

    Examples
    ========
    >>> foo = _get_peptide_ions("AA", [1,2])
    >>> foo
    {'z1y1': 90.054955167, ..., 'z2b1-NH3': 28.0125589145}
    >>> {k:v for k,v in foo.items() if "-" not in k} # This removes the neutral losses
    {'z1y1': 90.054955167, 'z1b1': 72.044390467, 'z2y1': 45.531115817, 'z2b1': 36.525833467}
    """
    fw, bw = get_forward_backward(aa_seq)
    out = {}

    for charge in charges:
        for ion in ion_types:
            ion_dict = get_annotation(fw, bw, charge, ion)
            ion_dict = {"z" + str(charge) + k: v for k, v in ion_dict.items()}
            out.update(ion_dict)

    return out


def get_tolerance(
    theoretical: float64, tolerance: int = 25, unit: str = "ppm"
) -> float64:
    if unit == "ppm":
        return theoretical * float(tolerance) / 10 ** 6
    elif unit == "da":
        return float(tolerance)
    else:
        raise ValueError("unit {} not implemented".format(unit))


def is_in_tolerance(
    theoretical: float64, observed: float, tolerance: int = 25, unit: str = "ppm"
) -> bool_:
    mz_tolerance = get_tolerance(theoretical, tolerance, unit)
    lower = observed - mz_tolerance
    upper = observed + mz_tolerance
    return theoretical >= lower and theoretical <= upper
