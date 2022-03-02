import random

from elfragmentador import encoding_decoding
from elfragmentador.annotate import get_peptide_ions
from elfragmentador.scoring import (
    calc_ascore,
    calc_delta_ascore,
    get_site_localizing_ions,
)
from elfragmentador.spectra import Spectrum


def test_ascore_calculation_works():
    seqs = [
        "M[OXIDATION]YPEPT[PHOSPHO]MIDES",
        "MYPEPT[PHOSPHO]M[OXIDATION]IDES",
        "MY[PHOSPHO]PEPTM[OXIDATION]IDES",
        "MYPEPTM[OXIDATION]IDES[PHOSPHO]",
    ]

    for seq in seqs:
        mzs = [i for i in get_peptide_ions(seq).values()]
        ints = [random.randint(0, 100) for _ in range(len(mzs))]
        len(mzs)
        len(ints)

        mods_list = ["PHOSPHO", "OXIDATION"]
        aas_list = ["STY", "M"]
        ascores = calc_ascore(seq, mods_list, aas_list, mzs, ints)
        calc_delta_ascore(seq, mods_list, aas_list, mzs, ints)

        assert ascores[seq] == max(
            ascores.values()
        ), "Max ascore does not correspond to the ground truth peptide"


def test_ascore_calculation_benchmark(benchmark):
    seq = "M[OXIDATION]YPEPT[PHOSPHO]MIDES"
    mods_list = ["PHOSPHO", "OXIDATION"]
    aas_list = ["STY", "M"]
    pep_ions = get_peptide_ions(seq)
    mzs = [i for i in pep_ions.values()]
    ints = [random.randint(0, 100) for _ in range(len(mzs))]

    benchmark(calc_delta_ascore, seq, mods_list, aas_list, mzs, ints)


def test_ascore_calculation_benchmark_many_posibilities(benchmark):
    seq = "M[OXIDATION]EQQPTRPPQTSQPPPPPPPM[OXIDATION]PFR"
    mods_list = ["OXIDATION"]
    aas_list = ["MP"]
    pep_ions = get_peptide_ions(seq)
    mzs = [i for i in pep_ions.values()]
    ints = [random.randint(0, 100) for _ in range(len(mzs))]
    benchmark(calc_delta_ascore, seq, mods_list, aas_list, mzs, ints)


def test_ascore_calculation_benchmark_nterm(benchmark):
    seq = "n[TMT6PLEX]M[OXIDATION]EQKPTRPPQTSQPKPPPPP[OXIDATION]PFR"
    mods_list = ["TMT6PLEX"]
    aas_list = ["nK"]
    pep_ions = get_peptide_ions(seq)
    mzs = [i for i in pep_ions.values()]
    ints = [random.randint(0, 100) for _ in range(len(mzs))]
    benchmark(calc_delta_ascore, seq, mods_list, aas_list, mzs, ints)


def test_testing_test():
    seq = "GHY[PHOSPHO]TIGK"
    seq2 = "GHYT[PHOSPHO]IGK"
    mods_list = ["PHOSPHO", "OXIDATION"]
    aas_list = ["STY", "M"]
    pep_ions = get_peptide_ions(seq)
    pep_ions2 = get_peptide_ions(seq2)
    mzs = [i for i in pep_ions.values()]
    mzs2 = [i for i in pep_ions2.values()]

    comb_mzs = []
    comb_mzs.extend(mzs)
    comb_mzs.extend(mzs2)
    comb_mzs = list(set(comb_mzs))

    ints = [1 for _ in range(len(mzs))]
    ints2 = [1 for _ in range(len(mzs2))]
    comb_ints = [1 for _ in range(len(comb_mzs))]

    o1, o2 = get_site_localizing_ions(seq, mods_list, aas_list)
    print(o1)

    out = calc_ascore(seq, mods_list, aas_list, mzs, ints)
    print(out)

    out = calc_ascore(seq, mods_list, aas_list, mzs2, ints2)
    print(out)

    out = calc_ascore(seq, mods_list, aas_list, comb_mzs, comb_ints)
    print(out)


def test_ascore_calculation_works_without_mods():
    seq = "MYPEPTMIDES"
    mods_list = ["PHOSPHO", "OXIDATION"]
    aas_list = ["STY", "M"]
    pep_ions = get_peptide_ions(seq)
    mzs = [i for i in pep_ions.values()]
    ints = [random.randint(0, 100) for _ in range(len(mzs))]

    out_ascore = calc_delta_ascore(seq, mods_list, aas_list, mzs, ints)

    assert out_ascore > 0
    print(out_ascore)


spectrum_1_string = """
Spectrum.from_tensors(
    [ 23, 10, 3, 10, 16, 16, 10, 1, 21, 16, 6, 9, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [ 0, 0.056, 0.855, 0.091, 0.251, 0.322, 0.129, 0.498, 0, 0.900, 0.052, 0.467, 0, 0.826, 0, 1.0, 0,
      0.365, 0, 0.209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.044, 0, 0.016, 0, 0.220, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    mod_tensor=[
        0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
)
"""

spectrum_2_string = """
Spectrum.from_tensors(
    [ 23, 10, 3, 10, 16, 16, 10, 1, 21, 16, 6, 9, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [ 0, 0.056, 0.855, 0.091, 0.251, 0.322, 0.129, 0.498, 0, 0.900, 0.052, 0.467, 0, 0.826, 0, 1.0, 0,
      0.365, 0, 0.209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.044, 0, 0.016, 0, 0.220, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    mod_tensor=[
        0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],)
"""


def test_ascore_ascore_calculates_correctly_best_annotation():
    test_spec1 = eval(spectrum_1_string)
    test_spec2 = eval(spectrum_2_string)

    print(f"{test_spec2.delta_ascore} < {test_spec1.delta_ascore}")
    assert test_spec2.delta_ascore < test_spec1.delta_ascore

    ascore_wrong = calc_delta_ascore(
        test_spec2.mod_sequence,
        ["PHOSPHO"],
        ["STY"],
        test_spec1.mzs,
        test_spec1.intensities,
    )
    assert ascore_wrong < 0
