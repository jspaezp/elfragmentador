"""
Greatly inspired/copied from:
https://github.com/kusterlab/prosit/blob/master/prosit

And released under an Apache 2.0 license
"""

import collections
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, List, Tuple, Union, Iterator
import warnings

import numpy
from numpy import bool_, float64, ndarray

from elfragmentador import constants, encoding_decoding


def solve_alias(x):
    try:
        x = x if len(x) == 1 else x[:1] + f"[{constants.MOD_PEPTIDE_ALIASES[x]}]"
    except KeyError:
        x = x if len(x) == 1 else x[:1] + f"[{constants.MOD_PEPTIDE_ALIASES[x.replace('[', '[+')]}]"

    x = x if len(x) != 3 else x[:1]  # Takes care of C[]

    return x


def peptide_parser(p: str, solve_aliases=False) -> Iterator[str]:
    """
    Parses maxquant formatted peptide strings

    Examples
    ========
    >>> list(peptide_parser("AAACC"))
    ['n', 'A', 'A', 'A', 'C', 'C', 'c']
    >>> list(peptide_parser("AAAM(ox)CC"))
    ['n', 'A', 'A', 'A', 'M(ox)', 'C', 'C', 'c']
    >>> list(peptide_parser("AAAM[+16]CC"))
    ['n', 'A', 'A', 'A', 'M[+16]', 'C', 'C', 'c']
    """

    ANNOTATIONS = "[](){}"

    if p[0] in ANNOTATIONS:
        raise ValueError(f"sequence starts with '{p[0]}'")
    n = len(p)
    i = 0

    # Yield n terminus if its not explicit in the sequence
    if p[0] != "n":
        yield "n"

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
            out = p[i:offset]
            yield_value = solve_alias(out) if solve_aliases else out
            i = offset
        else:
            yield_value = p[i]
            i += 1

        yield yield_value

    # Yield c terminus if its not explicit in the sequence
    if yield_value != "c":
        yield "c"


def mass_diff_encode_seq(seq):
    iter = peptide_parser(seq, solve_aliases=True)
    iter = encoding_decoding.clip_explicit_terminus(list(iter))
    # For some reason skyline detects T[80] but not T[+80] ...
    # And does not detect T[181] as a valid mod ...
    out = "".join([constants.MASS_DIFF_ALIASES_I[x].replace("+", "") for x in iter])
    return out


def canonicalize_seq(seq: str, robust: bool = False) -> str:
    """canonicalize_seq Solves all modification aliases in a sequence.

    Given a sequence, converts al supported modification aliases to the
    "canonical" version of them and returns the new version.

    Parameters
    ----------
    seq : str
        Modified peptide sequence, for example "PEPTIDE[+23]TIDE")
    robust : bool, optional
        Wether you want error to be silent and return none when they happen, by default False

    Returns
    -------
    str
        Same sequence as input but with all mod aliases replaced for the primary
        one in this package

    Raises
    ------
    e
        [description]
    """
    try:
        out = "".join(peptide_parser(seq, solve_aliases=True))
    except KeyError as e:
        out = None
        if not robust:
            warnings.warn(f"Non-supported sequence found in {seq}")
            raise e

    return out


def get_theoretical_mass(peptide: str):
    """
    Calculates the theoretical mass of a peptide

    Example
    -------
    >>> get_theoretical_mass("MYPEPTIDE")
    1093.4637787
    """
    aas = peptide_parser(peptide)
    out = sum([constants.MOD_AA_MASSES[a] for a in aas])
    return out


def get_precursor_mz(peptide: str, charge: int):
    """
    Calculates the theoretical mass/charge of a precursor peptide
    (assumes positive mode)

    Example
    -------
    >>> get_precursor_mz("MYPEPTIDE", 1)
    1094.471055167
    >>> get_precursor_mz("MYPEPTIDE", 2)
    547.739165817
    """
    return get_mz(get_theoretical_mass(peptide), 0, charge)


def get_forward_backward(peptide: str) -> Tuple[ndarray, ndarray]:
    """
    Calculates masses forward and backwards from aminoacid
    sequences

    Examples
    ========
    >>> get_forward_backward("AMC")
    (array([  1.00782503,  72.04493903, 203.08542403, 363.11607276,
           380.11881242]), array([ 17.00273967, 177.03338839, 308.07387339, 379.11098739,
           380.11881242]))
    >>> get_forward_backward("AM[147]C")
    (array([  1.00782503,  72.04493903, 219.08033403, 379.11098276,
           396.11372242]), array([ 17.00273967, 177.03338839, 324.06878339, 395.10589739,
           396.11372242]))
    >>> get_forward_backward("n[+42]AM[147]C")
    (array([ 43.01839004, 114.05550404, 261.09089904, 421.12154776,
           438.12428742]), array([ 17.00273967, 177.03338839, 324.06878339, 395.10589739,
           438.12428742]))
    """
    amino_acids = peptide_parser(peptide)
    masses = [constants.MOD_AA_MASSES[a] for a in amino_acids]
    forward = numpy.cumsum(masses)
    backward = numpy.cumsum(masses[::-1])
    return forward, backward


def get_mz(sum_: float64, ion_offset: float, charge: int) -> float64:
    return (sum_ + ion_offset + charge * constants.PROTON) / charge


def get_mzs(cumsum: ndarray, ion_type: str, z: int) -> List[float64]:
    # return (cumsum[:-1] + constants.ION_OFFSET[ion_type] + (z * constants.PROTON))/z
    return [get_mz(s, constants.ION_OFFSET[ion_type], z) for s in cumsum[:-2]][1:]


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
    array([  1.00782503,  72.04493903, 203.08542403, 363.11607276, 380.11881242])
    >>> bw
    array([ 17.00273967, 177.03338839, 308.07387339, 379.11098739, 380.11881242])
    >>> get_annotation(fw, bw, 3, "y")
    OrderedDict([('y1', 60.354347608199994), ...])
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
            raise ValueError("unknown ion_type: {}".format(ion_type))
        masses = get_mzs(cummass, ion_type, charge)
        d = {tmp.format(ion_type, i + 1): m for i, m in enumerate(masses)}
        all_.update(d)
        """
        for nl, offset in constants.NEUTRAL_LOSS.items():
            nl_masses = get_mzs(cummass - offset, ion_type, charge)
            d = {tmp_nl.format(ion_type, i + 1, nl): m for i, m in enumerate(nl_masses)}
            all_.update(d)
        """
    return collections.OrderedDict(sorted(all_.items(), key=lambda t: t[0]))


def get_peptide_ions(aa_seq: str) -> Dict[str, float64]:
    """Gets the theoretical masses of fragment ions

    Args:
        aa_seq (str): Aminoacid sequence with modifications

    Returns:
        Dict[str, float64]: Keys are ion names and values are the mass

    Examples:
    >>> foo = get_peptide_ions("AA")
    >>> sorted(foo.keys())
    ['z1b1', 'z1y1', 'z2b1', 'z2y1', 'z3b1', 'z3y1']
    >>> foo['z1y1'] # ground truth from http://db.systemsbiology.net:8080/proteomicsToolkit/FragIonServlet.html
    90.054955167
    >>> foo['z1b1']
    72.044390467
    """
    out = _get_peptide_ions(
        aa_seq,
        charges=range(1, constants.MAX_FRAG_CHARGE + 1),
        ion_types=constants.ION_TYPES,
    )
    return out


def _get_peptide_ions(
    aa_seq: str,
    charges: Union[List[int], range] = range(1, 5),
    ion_types: Union[str, List[str]] = "yb",
) -> Dict[str, float64]:
    """

    Examples
    ========
    >>> foo = _get_peptide_ions("AA", [1,2])
    >>> foo
    {'z1y1': 90.054955167, ...}
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


def annotate_peaks(theoretical_peaks, mzs, intensities, tolerance=25, unit="ppm"):
    annots = {}
    max_delta = tolerance if unit == "da" else max(mzs) * tolerance / 1e6

    raise DeprecationWarning(
        "Use `annotate_peaks2`, this version of the"
        " function is deprecated and will be removed in the future"
    )

    for mz, inten in zip(mzs, intensities):
        matching = {
            k: inten
            for k, v in theoretical_peaks.items()
            if abs(mz - v) <= max_delta and is_in_tolerance(v, mz, tolerance, unit)
        }
        annots.update(matching)

    max_int = max([v for v in annots.values()] + [0])
    annots = {k: v / max_int for k, v in annots.items()}
    return annots


def is_sorted(
    lst: Union[
        List[List[Union[int, float]]],
        List[List[float]],
        List[List[Union[str, float64]]],
        List[List[Union[float64, int]]],
    ],
    key: Callable = lambda x: x,
) -> bool:
    for i, el in enumerate(lst[1:]):
        if key(el) < key(lst[i]):  # i is the index of the previous element
            return False
    return True


def sort_if_needed(
    lst: Union[
        List[List[Union[int, float]]],
        List[List[float]],
        List[List[Union[str, float64]]],
        List[List[Union[float64, int]]],
    ],
    key: Callable = lambda x: x,
) -> None:
    if not is_sorted(lst, key):
        lst.sort(key=key)


def annotate_peaks2(
    theoretical_peaks: Dict[str, float64],
    mzs: Union[List[float64], List[float]],
    intensities: Union[List[float], List[int]],
    tolerance: int = 25,
    unit: str = "ppm",
) -> Dict[str, float]:
    max_delta = tolerance if unit == "da" else max(mzs) * tolerance / 1e6

    mz_pairs = [[m, i] for m, i in zip(mzs, intensities)]
    theo_peaks = [[k, v] for k, v in theoretical_peaks.items()]

    sort_if_needed(mz_pairs, key=lambda x: x[0])
    sort_if_needed(theo_peaks, key=lambda x: x[1])

    theo_iter = iter(theo_peaks)
    curr_theo_key, curr_theo_val = next(theo_iter)

    annots = defaultdict(lambda: 0)
    for mz, inten in mz_pairs:
        deltamass = mz - curr_theo_val
        try:
            while deltamass >= max_delta:
                curr_theo_key, curr_theo_val = next(theo_iter)
                deltamass = mz - curr_theo_val
        except StopIteration:
            pass

        in_deltam = abs(deltamass) <= max_delta
        if in_deltam and abs(deltamass) <= get_tolerance(
            curr_theo_val, tolerance, unit
        ):
            annots[curr_theo_key] += inten
    else:
        try:
            while True:
                curr_theo_key, curr_theo_val = next(theo_iter)
                deltamass = mz - curr_theo_val
                if deltamass < -max_delta:
                    break
                in_deltam = abs(deltamass) <= max_delta
                if in_deltam and abs(deltamass) <= get_tolerance(
                    curr_theo_val, tolerance, unit
                ):
                    annots[curr_theo_key] += inten
        except StopIteration:
            pass

    max_int = max([v for v in annots.values()] + [0])
    annots = {k: v / max_int for k, v in annots.items()}
    return annots
