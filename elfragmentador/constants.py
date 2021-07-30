"""
Defines the constants that are used in the rest of the project.

Such as the masses of aminoacids, supported modifications, length of the encodings,
maximum length supported, labels and order of the encoded ions ...

Greatly inspired/copied from:
https://github.com/kusterlab/prosit/blob/master/prosit/constants.py

And released under an Apache 2.0 license
"""

VAL_SPLIT = 0.8

TOLERANCE_FTMS = 25
TOLERANCE_ITMS = 0.35
TOLERANCE_TRIPLETOF = 0.5

TOLERANCE = {"FTMS": (25, "ppm"), "ITMS": (0.35, "da"), "TripleTOF": (50, "ppm")}

CHARGES = [1, 2, 3, 4, 5, 6]
DEFAULT_MAX_CHARGE = max(CHARGES)
MAX_FRAG_CHARGE = 3
MAX_SEQUENCE = 30
MAX_TENSOR_SEQUENCE = 30 + 2
MAX_ION = MAX_SEQUENCE - 1
ION_TYPES = ["y", "b"]
ION_TYPES = sorted(ION_TYPES)

NLOSSES = ["", "H2O", "NH3"]

FORWARD = {"a", "b", "c"}
BACKWARD = {"x", "y", "z"}

# Atomic elements
PROTON = 1.007276467
ELECTRON = 0.00054858
H = 1.007825035
C = 12.0
O = 15.99491463
N = 14.003074

# Tiny molecules
N_TERMINUS = H
C_TERMINUS = O + H
CO = C + O
CHO = C + H + O
NH2 = N + H * 2
H2O = H * 2 + O
NH3 = N + H * 3

NEUTRAL_LOSS = {"NH3": NH3, "H2O": H2O}

ION_OFFSET = {
    "a": 0 - CHO,
    "b": 0 - H,
    "c": 0 + NH2,
    "x": 0 + CO - H,
    "y": 0 + H,
    "z": 0 - NH2,
}

# Amino acids
# Modifications use high caps PSI-MS name
MODIFICATION = {
    "CARBAMIDOMETHYL": 57.0214637236,  # Carbamidomethylation (CAM)
    "ACETYL": 42.010565,  # Acetylation
    "DEAMIDATED": 0.984016,  # Deamidation
    "OXIDATION": 15.99491,  # Oxidation
    "PHOSPHO": 79.966331,  # Phosphorylation
    "METHYL": 14.015650,  # Methylation
    "DIMETHYL": 28.031300,  # Dimethylation
    "TRIMETHYL": 42.046950,  # Trimethylation
    "FORMYL": 27.994915,  # Formylation
    "GG": 114.042927,  # GlyGly ubiquitinylation residue
    "LRGG": 383.228103,  # LeuArgGlyGly ubiquitinylation residue
    "NITRO": 44.985078,  #  Oxidation to nitro
    "BIOTINYL": 226.077598,  #  Biotinilation
}

VARIABLE_MODS = {
    "ACETYL": "Kn",  # Acetylation
    "BIOTINYL": "K",  # Biotinilation
    "DEAMIDATED": "RNQ",  # Deamidation
    "OXIDATION": "MP",  # Oxidation
    "PHOSPHO": "STY",  # Phosphorylation
    "METHYL": "KR",  # Methylation
    "DIMETHYL": "KR",  # Dimethylation
    "TRIMETHYL": "K",  # Trimethylation
    "FORMYL": "K",  # Formylation
    "GG": "K",  # GlyGly ubiquitinylation residue
    "NITRO": "Y",  #  Oxidation to nitro
}

MOD_INDICES = {v: i + 1 for i, v in enumerate(MODIFICATION)}
# {'': 0, 'CARBAMIDOMETHYL': 1, 'ACETYL': 2, 'DEAMIDATED': 3, ...

MOD_INDICES_S = {integer: char for char, integer in MOD_INDICES.items()}
# {0: '', 1: 'CARBAMIDOMETHYL', 2: 'ACETYL',

AMINO_ACID = {
    "G": 57.021464,
    "R": 156.101111,
    "V": 99.068414,
    "P": 97.052764,
    "S": 87.032028,
    "U": 150.95363,
    "L": 113.084064,
    "M": 131.040485,
    "Q": 128.058578,
    "N": 114.042927,
    "Y": 163.063329,
    "E": 129.042593,
    "C": 103.009185 + MODIFICATION["CARBAMIDOMETHYL"],
    "F": 147.068414,
    "I": 113.084064,
    "A": 71.037114,
    "T": 101.047679,
    "W": 186.079313,
    "H": 137.058912,
    "D": 115.026943,
    "K": 128.094963,
    "n": N_TERMINUS,  # Placeholder to have n terminal modifications
    "c": C_TERMINUS,  # Placeholder to have c terminal modifications
}

AMINO_ACID_SET = set(AMINO_ACID)
ALPHABET = {v: i + 1 for i, v in enumerate(sorted(AMINO_ACID))}
# {'A': 1, 'C': 2, ... 'W': 20, 'Y': 21}

ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
# {1: 'A', 2: 'C', ..., 20: 'W', 21: 'Y'}

AAS_NUM = len(ALPHABET)

MOD_PEPTIDE_ALIASES = {
    "C[160]": "",  # This makes it so it assumes it is always modified
    "C[+57]": "",  # This makes it so it assumes it is always modified
    "M(ox)": "OXIDATION",
    "M[OXIDATION]": "OXIDATION",
    "P[OXIDATION]": "OXIDATION",  # Hydroxylation of proline
    "S[PHOSPHO]": "PHOSPHO",
    "Y[PHOSPHO]": "PHOSPHO",
    "S[PHOS]": "PHOSPHO",
    "T[PHOSPHO]": "PHOSPHO",
    "T[PHOS]": "PHOSPHO",
    "K[Acetyl]": "ACETYL",
    "K[GlyGly]": "GG",
    "K[156]": "FORMYL",  # or "DIMETHYL",
    "P[113]": "OXIDATION",  # aka hydroxilation
    "R[157]": "DEAMIDATED",  # aka citrullinated
    "n[43]": "ACETYL",  # n-terminal acetylation
    "n[ACETYL]": "ACETYL",  # n-terminal acetylation
}

# Adds the cannonical names to the aliases, like K[GG]
[
    MOD_PEPTIDE_ALIASES.update(
        {f"{mod_aa}[{mod_name}]": mod_name for mod_aa in mod_aminoacids}
    )
    for mod_name, mod_aminoacids in VARIABLE_MODS.items()
]

# This generages aliases like T[+80], M[+16.99], M[+16.9999]
int_aliases = []
for rounding_term in [0, 2, 4]:
    for k, v in VARIABLE_MODS.items():
        int_aliases.append(
            {
                aa + f"[+{round(MODIFICATION[k], rounding_term):.{rounding_term}f}]": k
                for aa in v
            }
        )

# This generates M[80] from M[+80]
MASS_DIFF_ALIASES = {}
_ = [MASS_DIFF_ALIASES.update(x) for x in int_aliases[::-1]]
MASS_DIFF_ALIASES_I = {k[0] + f"[{v}]": k for k, v in MASS_DIFF_ALIASES.items()}
MASS_DIFF_ALIASES_I.update({"C": "C[+57]"})
MASS_DIFF_ALIASES_I.update({k: k for k in AMINO_ACID})

MOD_PEPTIDE_ALIASES.update(MASS_DIFF_ALIASES)
# This generages aliases like T[181]
int_aliases = [
    {aa + f"[{str(round(MODIFICATION[k] + AMINO_ACID[aa]))}]": k for aa in v}
    for k, v in VARIABLE_MODS.items()
]
[MOD_PEPTIDE_ALIASES.update(x) for x in int_aliases[::-1]]
del int_aliases

MOD_AA_MASSES = AMINO_ACID.copy()
MOD_AA_MASSES.update(
    {
        k: AMINO_ACID[k[0]] + MODIFICATION.get(v, 0)
        for k, v in MOD_PEPTIDE_ALIASES.items()
    }
)


ION_ENCODING_NESTING = ["CHARGE", "POSITION", "ION_TYPE"]
ION_ENCODING_ITERABLES = {
    "ION_TYPE": "".join(sorted(ION_TYPES)),
    "CHARGE": [f"z{z}" for z in range(1, MAX_FRAG_CHARGE + 1)],
    "POSITION": list(range(1, MAX_ION + 1)),
}
FRAG_EMBEDING_LABELS = []

# TODO implement neutral losses ...  if needed
for charge in ION_ENCODING_ITERABLES[ION_ENCODING_NESTING[0]]:
    for pos in ION_ENCODING_ITERABLES[ION_ENCODING_NESTING[1]]:
        for ion in ION_ENCODING_ITERABLES[ION_ENCODING_NESTING[2]]:
            key = f"{charge}{ion}{pos}"
            FRAG_EMBEDING_LABELS.append(key)

NUM_FRAG_EMBEDINGS = len(FRAG_EMBEDING_LABELS)


IRT_PEPTIDES = {
    "LGGNEQVTR": {"vendor": "biognosys", "irt": -24.92},
    "GAGSSEPVTGLDAK": {"vendor": "biognosys", "irt": 0},
    "VEATFGVDESNAK": {"vendor": "biognosys", "irt": 12.39},
    "YILAGVENSK": {"vendor": "biognosys", "irt": 19.79},
    "TPVISGGPYEYR": {"vendor": "biognosys", "irt": 28.71},
    "TPVITGAPYEYR": {"vendor": "biognosys", "irt": 33.38},
    "DGLDAASYYAPVR": {"vendor": "biognosys", "irt": 42.26},
    "ADVTPADFSEWSK": {"vendor": "biognosys", "irt": 54.62},
    "GTFIIDPGGVIR": {"vendor": "biognosys", "irt": 70.52},
    "GTFIIDPAAVIR": {"vendor": "biognosys", "irt": 87.23},
    "LFLQFGAQGSPFLK": {"vendor": "biognosys", "irt": 100},
    "HEHISSDYAGK": {"vendor": "procal", "irt": -36.83},
    "IGYDHGHIEHK": {"vendor": "procal", "irt": -33.5},
    "TFAHTESHISK": {"vendor": "procal", "irt": -33.32},
    "ISLGEHEGGGK": {"vendor": "procal", "irt": -18.54},
    "YVGDSYDSSAK": {"vendor": "procal", "irt": -16.87},
    "FGTGTYAGGEK": {"vendor": "procal", "irt": -9.35},
    "LSSGYDGTSYK": {"vendor": "procal", "irt": -8.82},
    "TASGVGGFSTK": {"vendor": "procal", "irt": -4.18},
    "LTSGDFGEDSK": {"vendor": "procal", "irt": -3.76},
    "AGDEALGDTYK": {"vendor": "procal", "irt": -3.52},
    "SYASDFGSSAK": {"vendor": "procal", "irt": 1.79},
    "LYSYYSSTESK": {"vendor": "procal", "irt": 6.39},
    "FASDTSDEAFK": {"vendor": "procal", "irt": 7.2},
    "LTDTFADDDTK": {"vendor": "procal", "irt": 8.25},
    "LYTGAGYDEVK": {"vendor": "procal", "irt": 10.53},
    "TLIAYDDSSTK": {"vendor": "procal", "irt": 14.98},
    "TASEFDSAIAQDK": {"vendor": "procal", "irt": 17.84},
    "HDLDYGIDSYK": {"vendor": "procal", "irt": 19.86},
    "FLASSEGGFTK": {"vendor": "procal", "irt": 20.88},
    "HTAYSDFLSDK": {"vendor": "procal", "irt": 25.9},
    "FVGTEYDGLAK": {"vendor": "procal", "irt": 26.82},
    "YALDSYSLSSK": {"vendor": "procal", "irt": 32},
    "YYGTIEDTEFK": {"vendor": "procal", "irt": 33.73},
    "GFLDYESTGAK": {"vendor": "procal", "irt": 35.9},
    "HLTGLTFDTYK": {"vendor": "procal", "irt": 36.5},
    "YFGYTSDTFGK": {"vendor": "procal", "irt": 41.42},
    "HDTVFGSYLYK": {"vendor": "procal", "irt": 41.42},
    "FSYDGFEEDYK": {"vendor": "procal", "irt": 44.22},
    "ALFSSITDSEK": {"vendor": "procal", "irt": 44.88},
    "LYLSEYDTIGK": {"vendor": "procal", "irt": 48.16},
    "HFALFSTDVTK": {"vendor": "procal", "irt": 50.41},
    "VSGFSDISIYK": {"vendor": "procal", "irt": 51.67},
    "GSGGFTEFDLK": {"vendor": "procal", "irt": 51.97},
    "TFTGTTDSFFK": {"vendor": "procal", "irt": 52.2},
    "TFGTETFDTFK": {"vendor": "procal", "irt": 54.53},
    "YTSFYGAYFEK": {"vendor": "procal", "irt": 56.65},
    "LTDELLSEYYK": {"vendor": "procal", "irt": 57.66},
    "ASDLLSGYYIK": {"vendor": "procal", "irt": 57.68},
    "YGFSSEDIFTK": {"vendor": "procal", "irt": 57.77},
    "HTYDDEFFTFK": {"vendor": "procal", "irt": 58.44},
    "FLFTGYDTSVK": {"vendor": "procal", "irt": 61.07},
    "GLSDYLVSTVK": {"vendor": "procal", "irt": 61.34},
    "VYAETLSGFIK": {"vendor": "procal", "irt": 62.57},
    "GLFYGGYEFTK": {"vendor": "procal", "irt": 62.96},
    "GSTDDGFIILK": {"vendor": "procal", "irt": 63.07},
    "TSIDSFIDSYK": {"vendor": "procal", "irt": 63.51},
    "TLLLDAEGFEK": {"vendor": "procal", "irt": 65.49},
    "GFVIDDGLITK": {"vendor": "procal", "irt": 66.46},
    "GFEYSIDYFSK": {"vendor": "procal", "irt": 66.9},
    "GIFGAFTDDYK": {"vendor": "procal", "irt": 71.49},
    "LEIYTDFDAIK": {"vendor": "procal", "irt": 71.99},
    "FTEGGILDLYK": {"vendor": "procal", "irt": 72.95},
    "LLFSYSSGFVK": {"vendor": "procal", "irt": 73.23},
    "STFFSFGDVGK": {"vendor": "procal", "irt": 74.29},
    "LTAYFEDLELK": {"vendor": "procal", "irt": 75.09},
    "VDTFLDGFSVK": {"vendor": "procal", "irt": 76.57},
    "GASDFLSFAVK": {"vendor": "procal", "irt": 77.42},
    "GEDLDFIYVVK": {"vendor": "procal", "irt": 79.62},
    "VSSIFFDTFDK": {"vendor": "procal", "irt": 82.28},
    "SILDYVSLVEKK": {"vendor": "procal", "irt": 83.05},
    "VYGYELTSLFK": {"vendor": "procal", "irt": 87.89},
    "GGFFSFGDLTK": {"vendor": "procal", "irt": 88.04},
    "YDTAIDFGLFK": {"vendor": "procal", "irt": 89.4},
    "IVLFELEGITK": {"vendor": "procal", "irt": 94.97},
    "GIEDYYIFFAK": {"vendor": "procal", "irt": 95.37},
    "SILDYVSLVEK": {"vendor": "procal", "irt": 96.26},
    "AFSDEFSYFFK": {"vendor": "procal", "irt": 99.13},
    "AFLYEIIDIGK": {"vendor": "procal", "irt": 99.61},
}


del charge
del pos
del ion
del key

if __name__ == "__main__":
    # This is implemented so the constants can be printed if needed running this file directly
    my_vars = {k: v for k, v in globals().items() if not k.startswith("_")}
    for k, v in my_vars.items():
        print(f"\n>>> {k} {type(v)} = {v}")
