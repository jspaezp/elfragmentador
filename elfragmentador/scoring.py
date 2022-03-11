import math
from typing import Any, Dict, List, Tuple, Union

from numpy import float64

from elfragmentador import annotate, isoforms


def get_site_localizing_ions(
    seq: str, mod: List[str], aas: List[str]
) -> Union[
    Tuple[Dict[str, Dict[Any, Any]], Dict[str, Dict[str, float64]]],
    Tuple[Dict[str, Dict[str, float64]], Dict[str, Dict[str, float64]]],
]:
    """
    get_site_localizing_ions.

    [extended_summary]

    Parameters
    ----------
    seq : str
        [description]
    mod : List[str]
        [description]
    aas : List[str]
        [description]

    Returns
    -------
    Tuple[
        Dict[str, Dict[str, float64]],
        Dict[str, Dict[str, float64]]]

    Returns two dictionaries whose keys are isoform sequences and the values are
    dictionaries of ion:mass pairs.

    The first one contins only the ions unique to each sequence and the second contains
    all the ions for each modified sequence

    Example
    -------
    >>> seq = "MY[PHOSPHO]PTMIDE"
    >>> mods_list = ["PHOSPHO"]
    >>> aas_list = ["YST"]
    >>> sli = get_site_localizing_ions(seq, mods_list, aas_list)
    >>> sorted(sli[0].keys())
    ['MYPT[PHOSPHO]MIDE', 'MY[PHOSPHO]PTMIDE']
    >>> sorted(sli[0]['MYPT[PHOSPHO]MIDE'].keys())
    ['z1b2', 'z1b3', 'z1y5', 'z1y6', 'z2b2', 'z2b3', 'z2y5', 'z2y6', 'z3b2', \
        'z3b3', 'z3y5', 'z3y6']
    >>> sorted(sli[1].keys())
    ['MYPT[PHOSPHO]MIDE', 'MY[PHOSPHO]PTMIDE']
    >>> sorted(sli[1]['MYPT[PHOSPHO]MIDE'].keys())
    ['z1b1', 'z1b2', 'z1b3', ..., 'z1y6', 'z1y7', 'z2b1', 'z2b2', 'z2b3', \
        'z2b4', 'z2b5', 'z2b6', 'z2b7', 'z2y1', 'z2y2', 'z2y3', 'z2y4', \
        'z2y5', 'z2y6', 'z2y7', 'z3b1', 'z3b2', 'z3b3', 'z3b4', 'z3b5', \
        'z3b6', 'z3b7', 'z3y1', 'z3y2', 'z3y3', 'z3y4', \
        'z3y5', 'z3y6', 'z3y7']
    >>> sli[1]['MYPT[PHOSPHO]MIDE']['z1y6']
    785.278700167
    >>> out = get_site_localizing_ions(seq, mods_list, aas_list)
    >>> # Show the length of every sub-item
    >>> [{k: len(x[k]) for k in sorted(x)} for x in out]
    [{'MYPT[PHOSPHO]MIDE': 12, 'MY[PHOSPHO]PTMIDE': 12}, \
        {'MYPT[PHOSPHO]MIDE': 42, 'MY[PHOSPHO]PTMIDE': 42}]
    """
    mod_isoforms = isoforms.get_mod_isoforms(seq, mod, aas)
    mod_isoform_ions = {k: annotate.get_peptide_ions(k) for k in mod_isoforms}

    filtered_out_dict = {k: {} for k in mod_isoforms}
    out_dict = {k: {} for k in mod_isoforms}
    for ion in mod_isoform_ions[mod_isoforms[0]]:
        unique_vals = list(
            set([round(float(v[ion]), 10) for k, v in mod_isoform_ions.items()])
        )

        for mi in mod_isoforms:
            out_dict[mi].update({ion: mod_isoform_ions[mi][ion]})
            if len(unique_vals) > 1:
                filtered_out_dict[mi].update({ion: mod_isoform_ions[mi][ion]})

    return filtered_out_dict, out_dict


# TODO consider if this has to be a public API
def calc_ascore(
    seq: str,
    mod: List[str],
    aas: List[str],
    mzs: Union[List[float64], List[float]],
    ints: Union[List[float], List[int]],
) -> Dict[str, Union[float, int]]:
    WINDOW_SIZE = 100  # daltons
    # BINNING = WINDOW_SIZE / (constants.TOLERANCE_FTMS / (1e6) )
    # aka, how many peaks can fit in a 100da window
    # Theoretically this would be correct but leads to really high scores...
    BINNING = 100
    N_PER_WINDOW = list(range(1, 11, 1))

    sli, all_ions = get_site_localizing_ions(seq, mod, aas)
    max_mz = math.ceil(max(mzs) / WINDOW_SIZE) * WINDOW_SIZE
    norm_spectra = {str(x): [] for x in N_PER_WINDOW}

    for range_start in range(0, max_mz, WINDOW_SIZE):
        curr_mzs_pairs = [
            [m, i]
            for m, i in zip(mzs, ints)
            if m <= range_start + WINDOW_SIZE and m > range_start
        ]
        if len(curr_mzs_pairs) == 0:
            continue

        # from : https://stackoverflow.com/questions/13070461/
        sorted_indices = sorted(
            range(len(curr_mzs_pairs)), key=lambda x: curr_mzs_pairs[x][1]
        )
        for n in N_PER_WINDOW:
            keep_indices = sorted_indices[-n:]
            norm_spectra[str(n)].extend([curr_mzs_pairs[x][0] for x in keep_indices])

    # First pass finding best depth
    scores = {k: [] for k in norm_spectra}
    for norm_depth, norm_depth_spectra in norm_spectra.items():
        prior = int(norm_depth) / BINNING
        norm_depth_spectra_ints = [1 for _ in range(len(norm_depth_spectra))]
        tmp_scores = _calculate_scores_dict(
            norm_depth_spectra, norm_depth_spectra_ints, all_ions, prior
        )
        scores[norm_depth].extend(list(tmp_scores.values()))

    try:
        # define best as highest delta score
        deltascores = {k: sorted(v)[-1] - sorted(v)[-2] for k, v in scores.items()}
        # define best as highest cumulative score
        # deltascores = {k: sum(v) for k, v in scores.items()}
    except IndexError:
        return {seq: max([x[0] for x in scores.values()]), "": 0}

    deltascores
    max_deltascores = max(deltascores.values())
    best_depth = [k for k, v in deltascores.items() if v == max_deltascores][0]
    best_depth

    # second pass finding score
    prior = int(best_depth) / BINNING
    norm_depth_spectra = norm_spectra[best_depth]
    norm_depth_spectra_ints = [1 for _ in range(len(norm_depth_spectra))]
    scores = {k: None for k in sli}

    scores = _calculate_scores_dict(
        norm_depth_spectra, norm_depth_spectra_ints, sli, prior
    )
    return scores


def calc_delta_ascore(
    seq: str,
    mod: List[str],
    aas: List[str],
    mzs: Union[List[float64], List[float]],
    ints: Union[List[float], List[int]],
) -> float:
    ascores = calc_ascore(seq, mod, aas, mzs, ints)
    seq_score = ascores.pop(seq)
    return seq_score - max(ascores.values())


def _calculate_scores_dict(
    mzs: Union[List[float64], List[float]],
    ints: List[int],
    ions_dict: Dict[str, Dict[str, float64]],
    prior: float,
) -> Dict[str, float]:
    EPSILON = 1e-31

    scores = {k: None for k in ions_dict}
    for isoform, theo_peaks_isoform in ions_dict.items():
        num_tot_peaks = len(theo_peaks_isoform)
        matched_peaks = sum(
            annotate.annotate_peaks(
                theo_peaks_isoform,
                mzs=mzs,
                intensities=ints,
            ).values()
        )
        unmatched_peaks = num_tot_peaks - matched_peaks
        prob = (prior ** matched_peaks) * ((1 - prior) ** unmatched_peaks)
        score = -10 * math.log10(prob + EPSILON)
        scores[isoform] = score
    return scores
