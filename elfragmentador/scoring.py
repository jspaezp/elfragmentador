import math
from elfragmentador import annotate
from elfragmentador import isoforms


def get_site_localizing_ions(seq, mod, aas):
    mod_isoforms = isoforms.get_mod_isoforms(seq, mod, aas)
    mod_isoform_ions = {k: annotate.get_peptide_ions(k) for k in mod_isoforms}

    filtered_out_dict = {k: {} for k in mod_isoforms}
    out_dict = {k: {} for k in mod_isoforms}
    for ion in mod_isoform_ions[mod_isoforms[0]]:
        unique_vals = list(
            set([round(v[ion], 10) for k, v in mod_isoform_ions.items()])
        )

        for mi in mod_isoforms:
            out_dict[mi].update({ion: mod_isoform_ions[mi][ion]})
            if len(unique_vals) > 1:
                filtered_out_dict[mi].update({ion: mod_isoform_ions[mi][ion]})

    return filtered_out_dict, out_dict


def calc_ascore(seq, mod, aas, mzs, ints):
    sli, all_ions = get_site_localizing_ions(seq, mod, aas)
    max_mz = math.ceil(max(mzs) / 100) * 100
    n_per_100 = list(range(1, 11, 1))
    norm_spectra = {str(x): [] for x in n_per_100}

    for range_start in range(0, max_mz, 100):
        curr_mzs_pairs = [
            [m, i]
            for m, i in zip(mzs, ints)
            if m <= range_start + 100 and m > range_start
        ]
        if len(curr_mzs_pairs) == 0:
            continue

        # from : https://stackoverflow.com/questions/13070461/
        sorted_indices = sorted(
            range(len(curr_mzs_pairs)), key=lambda x: curr_mzs_pairs[x][1]
        )
        for n in n_per_100:
            keep_indices = sorted_indices[-n:]
            norm_spectra[str(n)].extend([curr_mzs_pairs[x][0] for x in keep_indices])

    # First pass finding best depth
    scores = {k: [] for k in norm_spectra}
    for norm_depth, norm_depth_spectra in norm_spectra.items():
        prior = int(norm_depth) / 100
        norm_depth_spectra_ints = [1 for _ in range(len(norm_depth_spectra))]
        tmp_scores = _calculate_scores_dict(
            norm_depth_spectra, norm_depth_spectra_ints, all_ions, prior
        )
        scores[norm_depth].extend(list(tmp_scores.values()))

    deltascores = {k: sorted(v)[-1] - sorted(v)[-2] for k, v in scores.items()}
    deltascores
    max_deltascores = max(deltascores.values())
    best_depth = [k for k, v in deltascores.items() if v == max_deltascores][0]
    best_depth

    # second pass finding score
    prior = int(best_depth) / 100
    norm_depth_spectra = norm_spectra[best_depth]
    norm_depth_spectra_ints = [1 for _ in range(len(norm_depth_spectra))]
    scores = {k: None for k in sli}

    scores = _calculate_scores_dict(
        norm_depth_spectra, norm_depth_spectra_ints, sli, prior
    )
    return scores


def calc_delta_ascore(seq, mod, aas, mzs, ints):
    ascores = calc_ascore(seq, mod, aas, mzs, ints)
    seq_score = ascores.pop(seq)
    return seq_score - max(ascores.values())


def _calculate_scores_dict(mzs, ints, ions_dict, prior):
    scores = {k: None for k in ions_dict}
    for isoform, theo_peaks_isoform in ions_dict.items():
        num_tot_peaks = len(theo_peaks_isoform)
        matched_peaks = sum(
            annotate.annotate_peaks2(
                theo_peaks_isoform,
                mzs=mzs,
                intensities=ints,
            ).values()
        )
        unmatched_peaks = num_tot_peaks - matched_peaks
        prob = (prior ** matched_peaks) * ((1 - prior) ** unmatched_peaks)
        score = -10 * math.log(prob)
        scores[isoform] = score
    return scores
