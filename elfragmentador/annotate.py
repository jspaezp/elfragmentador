"""
> Greatly inspired/copied from:

> https://github.com/kusterlab/prosit/blob/master/prosit
> And released under an Apache 2.0 license
"""

import warnings
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Tuple, Union

import numpy
import numpy as np
from numpy import bool_, float64, ndarray

from elfragmentador import constants, encoding_decoding


def _solve_alias(x: str) -> str:
    """
    _solve_alias Gets the cannnonical form of an aminoacid.

    Args:
        x (str): An aminoacid (possibly modified) to be queried

    Returns:
        str: Solved alias

    Examples:
        >>> _solve_alias("M(ox)")
        'M[OXIDATION]'
        >>> _solve_alias("M[+16]")
        'M[OXIDATION]'
        >>> _solve_alias("C[+57]")
        'C'
        >>> _solve_alias("C[Carbamidomethyl (C)]")
        'C'
    """
    try:
        x = x if len(x) == 1 else x[:1] + f"[{constants.MOD_PEPTIDE_ALIASES[x]}]"
    except KeyError as e:
        if len(x) > 1 and x[1] == "[" and x[2] in "1234567890":
            x = x[:1] + f"[{constants.MOD_PEPTIDE_ALIASES[x.replace('[', '[+')]}]"

        elif " " in x and x[-1] == "]" and x[1] == "[":
            content = x[2:-1].split(" ")[0].upper()
            query = f"{x[0]}[{content}]"
            x = x[0] + f"[{constants.MOD_PEPTIDE_ALIASES[query]}]"

        elif len(x) == 3:
            pass
        else:
            raise KeyError(e)

    x = x if len(x) != 3 else x[:1]  # Takes care of C[]

    return x


def peptide_parser(p: str, solve_aliases: bool = False) -> Iterator[str]:
    """
    peptide_parser Parses peptides in a string to an iterable.

    Args:
        p (str):
            Peptide sequence in a single string
        solve_aliases (bool, optional):
            Wether to solve aliases for modifications. Defaults to False.

    Raises:
        ValueError: Raises an error when the sequence cannot be correcly
            parsed (starts with a special chatacter for instance)

    Yields:
        Iterator[str]: Every element in the peptide sequence

    Examples:
        >>> list(peptide_parser("AAACC"))
        ['n', 'A', 'A', 'A', 'C', 'C', 'c']
        >>> list(peptide_parser("AAAM(ox)CC"))
        ['n', 'A', 'A', 'A', 'M(ox)', 'C', 'C', 'c']
        >>> list(peptide_parser("AAAM[+16]CC"))
        ['n', 'A', 'A', 'A', 'M[+16]', 'C', 'C', 'c']
        >>> list(peptide_parser("K.AAAM[+16]CC.K"))
        ['n', 'A', 'A', 'A', 'M[+16]', 'C', 'C', 'c']
        >>> list(peptide_parser("_KC[Carbamidomethyl (C)]W_"))
        ['n', 'K', 'C[Carbamidomethyl (C)]', 'W', 'c']
    """

    ANNOTATIONS = "[](){}"
    ANNOTATION_CLOSING = {"[": "]", "{": "}", "(": ")"}

    # This fixes a bug where passing a list would yield the incorrect results
    p = "".join(p)

    if p[1] == "." and p[-2] == ".":
        p = p[2:-2]

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
            closing_annotation = ANNOTATION_CLOSING[p[i + 1]]
            p_ = p[i + 2 :]
            j = p_.index(closing_annotation)
            offset = i + j + 3
            out = p[i:offset]
            try:
                yield_value = _solve_alias(out) if solve_aliases else out
            except KeyError:
                raise ValueError(f"Unable to Solve alias for {out}, in peptide {p}")

            i = offset
        else:
            yield_value = p[i]
            i += 1

        yield yield_value

    # Yield c terminus if its not explicit in the sequence
    if yield_value != "c":
        yield "c"


def mass_diff_encode_seq(seq: str) -> str:
    """
    Solve peptide string so modifications are expressed as mass difference
    without the +

    "T[+80]" > "T[80]"

    Args:
       seq (str): Sequence to convert

    Returns:
       str, Sequence with solved aliases
    """
    iter = peptide_parser(seq, solve_aliases=True)
    iter = encoding_decoding.clip_explicit_terminus(list(iter))
    # For some reason skyline detects T[80] but not T[+80] ...
    # And does not detect T[181] as a valid mod ...
    out = "".join([constants.MASS_DIFF_ALIASES_I[x].replace("+", "") for x in iter])
    return out


def canonicalize_seq(seq: str, robust: bool = False) -> str:
    """
    canonicalize_seq Solves all modification aliases in a sequence.

    Given a sequence, converts al supported modification aliases to the
    "canonical" version of them and returns the new version.

    Args:
      seq (str):
          Modified peptide sequence,
          for example "PEPTIDE[+23]TIDE")
      robust (bool):
          Wether you want error to be silent and return none
          when they happen, by default False

    Returns:
      str: Same sequence as input but with all mod aliases replaced for the primary
      one in this package

    Examples:
      >>> canonicalize_seq("PEPTM(ox)IDEPINK")
      'nPEPTM[OXIDATION]IDEPINKc'
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
    Calculates the theoretical mass of a peptide.

    Examples:
        >>> get_theoretical_mass("MYPEPTIDE")
        1093.4637787
    """
    aas = peptide_parser(peptide, solve_aliases=True)
    out = sum([constants.MOD_AA_MASSES[a] for a in aas])
    return out


def get_precursor_mz(peptide: str, charge: int):
    """
    Calculates the theoretical mass/charge of a precursor peptide (assumes
    positive mode)

    Args:
      peptide (str): Peptide string
      charge (int): charge of the peptide (positive integet)

    Examples:
        >>> get_precursor_mz("MYPEPTIDE", 1)
        1094.471055167
        >>> get_precursor_mz("MYPEPTIDE", 2)
        547.739165817
    """

    return (get_theoretical_mass(peptide) + charge * constants.PROTON) / charge


def _get_forward_backward(peptide: str) -> Tuple[ndarray, ndarray]:
    """
    Calculates masses forward and backwards from aminoacid sequences.

    Examples:
        >>> _get_forward_backward("AMC")
        (array([  1.00782503,  72.04493903, 203.08542403, 363.11607276,
            380.11881242]), array([ 17.00273967, 177.03338839,
            308.07387339, 379.11098739, 380.11881242]))
        >>> _get_forward_backward("AM[147]C")
        (array([  1.00782503,  72.04493903, 219.08033403, 379.11098276,
            396.11372242]), array([ 17.00273967, 177.03338839,
            324.06878339, 395.10589739, 396.11372242]))
        >>> _get_forward_backward("n[+42]AM[147]C")
        (array([ 43.01839004, 114.05550404, 261.09089904, 421.12154776,
            438.12428742]), array([ 17.00273967, 177.03338839,
            324.06878339, 395.10589739, 438.12428742]))
    """
    amino_acids = peptide_parser(peptide)
    masses = np.float64([constants.MOD_AA_MASSES[a] for a in amino_acids])
    forward = numpy.cumsum(masses)
    backward = numpy.cumsum(masses[::-1])
    return forward, backward


def _get_mzs(cumsum: ndarray, ion_type: str, z: int) -> np.float64:
    """
    Gets the m/z values from a series after being provided with the cumulative
    sums of the aminoacids in its series, meant for internal use.
    """

    cumsum = cumsum[:-2]
    out = (cumsum + constants.ION_OFFSET[ion_type] + z * constants.PROTON) / z
    out = out[1:]
    return out


def _get_annotation(
    forward: ndarray, backward: ndarray, charge: int, ion_types: str
) -> Dict[str, float]:
    """
    Calculates the ion annotations based on the forward and backward cumulative
    masses.

    Args:
      forward (ndarray): Forward cumulative mass
      backward (ndarray): Backwards cumulative mass
      charge (int): charge of the series to use
      ion_types (str): Ion times to calculate

    Returns:

    Example:
        >>> fw, bw  = _get_forward_backward("AMC")
        >>> fw
        array([  1.00782503,  72.04493903, 203.08542403, 363.11607276, 380.11881242])
        >>> bw
        array([ 17.00273967, 177.03338839, 308.07387339, 379.11098739, 380.11881242])
        >>> out = _get_annotation(fw, bw, 3, "y")
        >>> {k:round(float(v), 7) for k, v in out.items()}
        {'y1': 60.3543476, 'y2': 104.0345093}
    """
    all_ = {}
    for ion_type in ion_types:
        if ion_type in constants.FORWARD:
            cummass = forward
        elif ion_type in constants.BACKWARD:
            cummass = backward
        else:
            raise ValueError("unknown ion_type: {}".format(ion_type))
        masses = _get_mzs(cummass, ion_type, charge)
        d = {ion_type + str(i + 1): m for i, m in enumerate(np.nditer(masses))}
        all_.update(d)

    return all_


def get_peptide_ions(aa_seq: str) -> Dict[str, float64]:
    """
    Gets the theoretical masses of fragment ions.

    Args:
      aa_seq (str): Aminoacid sequence with modifications

    Returns:
      Dict[str, float64]: Keys are ion names and values are the mass
      Examples:

    Examples:
        >>> foo = get_peptide_ions("AA")
        >>> sorted(foo.keys())
        ['z1b1', 'z1y1', 'z2b1', 'z2y1', 'z3b1', 'z3y1']
        >>> # ground truth from
        >>> # http://db.systemsbiology.net:8080/proteomicsToolkit/FragIonServlet.html
        >>> print(round(foo['z1y1'], 6))
        90.054955
        >>> print(round(foo['z1b1'], 6))
        72.04439
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
    Gets a dictionary of theoretical ion masses for a peptide.

    Args:
        aa_seq: (str):
        charges: (Union[List[int], range]):
            Range of charges to use (Default value = range(1,5)
        ion_types: Union[str, List[str]]: Ion types to calculate (Default value = "yb")

    Returns:

    Examples:
        >>> foo = _get_peptide_ions("AA", [1,2])
        >>> {k:round(v, 8) for k, v in foo.items()}
        {'z1y1': 90.05495517, ...}
    """
    fw, bw = _get_forward_backward(aa_seq)
    out = {}

    for charge in charges:
        for ion in ion_types:
            ion_dict = _get_annotation(fw, bw, float(charge), ion)
            ion_dict = {"z" + str(charge) + k: float(v) for k, v in ion_dict.items()}
            out.update(ion_dict)

    return out


def get_tolerance(
    theoretical: float64, tolerance: Union[float, int] = 25.0, unit: str = "ppm"
) -> float64:
    """
    Calculates the toleranc in daltons from either a dalton tolerance or a ppm
    tolerance.

    Args:
      theoretical (float64): Theoretical m/z to be used (only used for ppm)
      tolerance (Union[float,int]): Tolerance value to be used (Default value = 25)
      unit (str): Lietrally da for daltons or ppm for ... ppm (Default value = "ppm")

    Returns:
        float, the tolerance value in daltons
    """
    if unit == "ppm":
        return theoretical * float(tolerance) / 10 ** 6
    elif unit == "da":
        return float(tolerance)
    else:
        raise ValueError("unit {} not implemented".format(unit))


def is_in_tolerance(
    theoretical: float64, observed: float, tolerance: int = 25, unit: str = "ppm"
) -> bool_:
    """
    Checks wether an observed mass is close enough to a theoretical mass.

    Args:
      theoretical (float64): Theoretical mass
      observed (float): Observed mass
      tolerance (int): Tolerance (Default value = 25)
      unit (str): Tolerance Unit (Default value = "ppm")

    Returns:
        bool, Wether the value observed is within the tolerance of the theoretical value
    """
    mz_tolerance = get_tolerance(theoretical, tolerance, unit)
    lower = observed - mz_tolerance
    upper = observed + mz_tolerance
    return theoretical >= lower and theoretical <= upper


def is_sorted(
    lst: Union[
        List[List[Union[int, float]]],
        List[List[float]],
        List[List[Union[str, float64]]],
        List[List[Union[float64, int]]],
    ],
    key: Callable = lambda x: x,
) -> bool:
    """
    is_sorted Checks if a list is sorted.

    Args:
        lst (List): List to check if it is sorted
        key (Callable, optional):
            Function to use as the key to compare.
            Defaults to lambda x:x.

    Returns:
        bool: Wether at least 1 element is out of order

    Examples:
        >>> is_sorted([1,2,3,4])
        True
        >>> is_sorted([1,2,2,3])
        True
        >>> is_sorted([4,2,2,3])
        False
    """
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
    """
    sort_if_needed Sorts a list in place if it is not already sorted.

    Args:
        lst (List): List to be sorted
        key (Callable, optional): Function to use as the key for sorting.
            Defaults to lambda x:x.

    Examples:
        >>> foo = [1,16,3,4]
        >>> sort_if_needed(foo)
        >>> foo
        [1, 3, 4, 16]
    """
    if not is_sorted(lst, key):
        lst.sort(key=key)


def annotate_peaks(
    theoretical_peaks: Dict[str, float64],
    mzs: Union[List[float64], List[float]],
    intensities: Union[List[float], List[int]],
    tolerance: int = 25,
    unit: str = "ppm",
) -> Dict[str, float]:
    """
    annotate_peaks Assigns m/z peaks to annotations.

    Args:
        theoretical_peaks (Dict[str, float64]):
            Dictionary specifying the names and masses of theoretical peaks
        mzs (Union[List[float64], List[float]]):
            Array of the masses to be annotated.
        intensities (Union[List[float], List[int]]):
            Array of the intensities that match the masses provided
        tolerance (int, optional):
            Tolerance to be used to count an observed and a theoretical m/z as a match.
            Defaults to 25.
        unit (str, optional):
            The unit of the formerly specified tolerance (da or ppm).
            Defaults to "ppm".

    Returns:
        Dict[str, float]:
            A dictionary with the keys being the names of the ions and the values being
            the intensities that were asigned to such ion.
    """
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
