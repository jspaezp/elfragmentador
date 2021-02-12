import warnings
from typing import List
from itertools import permutations
from elfragmentador.annotate import peptide_parser


# Answer from https://stackoverflow.com/questions/6284396
class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


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
        perm_iter = list(perm_unique(mod_sampler))

    out_seqs = []

    for i, x in enumerate(perm_iter):
        out_seqs.append(placeholder_seq.format(*x))

    return list(set(out_seqs))


def get_mod_isoforms(seq: str, mods_list: List[str], aas_list: List[str]) -> List[str]:
    # seq = "M[OXIDATION]YPEPT[PHOSPHO]MIDES"
    # mods_list = ["PHOSPHO", "OXIDATION"]
    # aas_list = ["STY", "M"]
    seqs = [seq]

    for mod, aas in zip(mods_list, aas_list):
        tmp_seqs = []
        for s in seqs:
            x = _get_mod_isoforms(s, mod, aas)
            tmp_seqs.extend(list(set(x)))
            if len(tmp_seqs) > 10000:
                warnings.warn("Large number of mod combinations found, clipping at 1k")
                continue

        seqs.extend(tmp_seqs)
    seqs.extend([seq])

    return list(set(seqs))
