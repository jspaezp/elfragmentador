import random
from elfragmentador.annotate import get_peptide_ions
from elfragmentador.scoring import calc_ascore, calc_delta_ascore


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
    print(pep_ions)
    mzs = [i for i in pep_ions.values()]
    ints = [random.randint(0, 100) for _ in range(len(mzs))]

    benchmark(calc_delta_ascore, seq, mods_list, aas_list, mzs, ints)
