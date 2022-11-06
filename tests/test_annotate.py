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


# Ground truth as defined in
# http://db.systemsbiology.net:8080/proteomicsToolkit/FragIonServlet.html
parametrized_peptides = (
    (
        "PEPICNK",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "PEPIC[160]NK",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "PEPIC[+57]NK",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "__PEPIC[+57]NK__",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "nPEPIC[+57]NKc",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "n[ACETYL]PEPIC[+57]NK",
        {"z1b6": 753.32364, "z1b4": 479.25007, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
)


@pytest.mark.parametrize(
    "peptide_string,ions_dict",
    parametrized_peptides,
)
def test_correct_masses_on_ions(peptide_string, ions_dict):
    # Note if not passing the parser, alias will not be searched correctly
    ions = annotate.get_peptide_ions(list(annotate.peptide_parser(peptide_string)))
    ions_joint = annotate.get_peptide_ions(
        "".join(annotate.peptide_parser(peptide_string))
    )
    for ion_name, ion_mass in ions_dict.items():
        assert abs(ions[ion_name] - ion_mass) < 0.001, {ion_name: ion_mass}
        assert abs(ions_joint[ion_name] - ion_mass) < 0.001, {ion_name: ion_mass}


def test_time_getting_annotations(benchmark):
    benchmark(annotate.get_peptide_ions, "SMERANDMPEPTIDE")
