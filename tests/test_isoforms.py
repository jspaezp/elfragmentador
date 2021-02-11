from elfragmentador.isoforms import get_mod_isoforms


def test_mod_isoforms_gets_all():
    out = get_mod_isoforms("S[PHOSPHO]AS", mods_list=["PHOSPHO"], aas_list=["STY"])
    print(out)
    expected = ["SAS[PHOSPHO]", "S[PHOSPHO]AS"]
    assert all(
        [x in out for x in expected]
    ), "Not all expected phosphoisoforms are detected"


def test_mod_isoform_works_on_multiple_aas():
    seq = "MYPEPT[PHOSPHO]IDES"
    out = get_mod_isoforms(seq, ["PHOSPHO"], ["STY"])
    print(out)
    expected = ["MY[PHOSPHO]PEPTIDES", "MYPEPT[PHOSPHO]IDES", "MYPEPTIDES[PHOSPHO]"]
    assert all(
        [x in out for x in expected]
    ), "Not all expected phosphoisoforms are detected"
    assert len(out) == len(expected), "Returning either more or less phosphoisoforms"


def test_mod_isoform_works_on_multiple_mods():
    seq = "M[OXIDATION]YPEPT[PHOSPHO]MIDES"
    mods_list = ["PHOSPHO", "OXIDATION"]
    aas_list = ["STY", "M"]
    out = get_mod_isoforms(seq, mods_list, aas_list)
    print(out)
    expected = [
        "M[OXIDATION]Y[PHOSPHO]PEPTMIDES",
        "M[OXIDATION]YPEPT[PHOSPHO]MIDES",
        "M[OXIDATION]YPEPTMIDES[PHOSPHO]",
        "MY[PHOSPHO]PEPTM[OXIDATION]IDES",
        "MYPEPT[PHOSPHO]M[OXIDATION]IDES",
        "MYPEPTM[OXIDATION]IDES[PHOSPHO]",
    ]
    assert all(
        [x in out for x in expected]
    ), "Not all expected phosphoisoforms are detected"
    assert len(out) == len(expected), "Returning either more or less phosphoisoforms"
