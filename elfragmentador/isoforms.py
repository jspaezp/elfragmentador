from typing import List
from itertools import permutations
from elfragmentador.annotate import peptide_parser


def _get_mod_isoforms(seq: str, mod: str, aas: str) -> List[str]:
    # mod = "PHOSPHO"
    # seq = "S[PHOSPHO]AS"
    # aas = "STY"
    if mod not in seq:
        return [seq]

    parsed_seq = list(peptide_parser(seq))
    stripped_seq = [x.replace(f"[{mod}]", "") for x in parsed_seq]

    placeholder_seq = [
        x if not any([x[:1] == y for y in aas]) else x[:1] + "{}" for x in stripped_seq
    ]
    placeholder_seq = "".join(placeholder_seq)
    mod_sampler = [x[1:] for x in parsed_seq if any([x[:1] == y for y in aas])]

    if len(set(mod_sampler)) == 1:
        perm_iter = [mod_sampler]
    else:
        perm_iter = list(set(permutations(mod_sampler, len(mod_sampler))))

    out_seqs = []

    for i, x in enumerate(perm_iter):
        out_seqs.append(placeholder_seq.format(*x))
        if i > 1000:
            print(mod_sampler)
            raise ValueError

    return list(set(out_seqs))


def get_mod_isoforms(seq: str, mods_list: List[str], aas_list: List[str]) -> List[str]:
    # seq = "M[OXIDATION]YPEPT[PHOSPHO]MIDES"
    # mods_list = ["PHOSPHO", "OXIDATION"]
    # aas_list = ["STY", "M"]
    seqs = [seq]

    for mod, aas in zip(mods_list, aas_list):
        tmp_seqs = [_get_mod_isoforms(x, mod, aas) for x in seqs]
        seqs = []
        [seqs.extend(x) for x in tmp_seqs]

    return list(set(seqs))
