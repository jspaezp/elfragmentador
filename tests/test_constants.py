from elfragmentador import constants


def test_all_modification_aliases_map():
    for v in constants.MOD_PEPTIDE_ALIASES.values():
        if v == "":
            continue
        constants.MODIFICATION[v]
