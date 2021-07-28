import pytest
from elfragmentador import annotate


def test_peptide_parser():
    print(list(annotate.peptide_parser("AAACC")))
    print(list(annotate.peptide_parser("AAA[+53]CC")))
    print(list(annotate.peptide_parser("AAA[+53]CC[+54]")))
    print(list(annotate.peptide_parser("__AAA[+53]CC[+54]__")))
    print(list(annotate.peptide_parser("__AAA[53]CC[54]__")))


def test_getting_ions():
    out = annotate.get_peptide_ions("MPEP")
    print(out)

    with pytest.raises(KeyError) as _:
        # This makes sure that there is no y4 ion reported for a
        # peptide of length 4
        out["z1y4"]

    rounded_out = [int(x) for x in out.values()]
    expected_mpep = [
        132,
        229,
        358,
        116,
        245,
        342,
        66,
        115,
        179,
        58,
        123,
        171,
        44,
        77,
        120,
        39,
        82,
        114,
    ]

    assert all([abs(x - y) <= 1 for x, y in zip(expected_mpep, rounded_out)])


def test_time_getting_annotations(benchmark):
    benchmark(annotate.get_peptide_ions, "SMERANDMPEPTIDE")
