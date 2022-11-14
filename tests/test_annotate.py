import pytest
from ms2ml import Peptide


def test_getting_ions():
    out = Peptide.from_proforma_seq("MPEP")
    out = out.ion_dict

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
        "[U:1]-PEPIC[U:4]NK",
        {"z1b6": 753.32364, "z1b4": 479.25007, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
)


@pytest.mark.parametrize(
    "peptide_string,ions_dict",
    parametrized_peptides,
)
def test_correct_masses_on_ions(peptide_string, ions_dict):
    # Note if not passing the parser, alias will not be searched correctly
    pep = Peptide.from_sequence(peptide_string)
    ions = pep.ion_dict
    for ion_name, ion_mass in ions_dict.items():
        assert abs(ions[ion_name] - ion_mass) < 0.001, {ion_name: ion_mass}
