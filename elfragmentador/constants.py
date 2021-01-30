"""
Defines the constants that are used in the rest of the project.

Such as the masses of aminoacids, supported modifications, length of the encodings,
maximum length supported, labes and order of the encoded ions ...

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
MAX_ION = MAX_SEQUENCE - 1
ION_TYPES = ["y", "b"]
ION_TYPES = sorted(ION_TYPES)

NLOSSES = ["", "H2O", "NH3"]

FORWARD = {"a", "b", "c"}
BACKWARD = {"x", "y", "z"}

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
}

AMINO_ACID_SET = set(AMINO_ACID)
ALPHABET = {v: i + 1 for i, v in enumerate(sorted(AMINO_ACID))}
# {'A': 1, 'C': 2, ... 'W': 20, 'Y': 21}

ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
# {1: 'A', 2: 'C', ..., 20: 'W', 21: 'Y'}

AAS_NUM = len(ALPHABET)

MOD_PEPTIDE_ALIASES = {
    "C[160]": "",  # This makes it so it assumes it is always modified
    "M[147]": "OXIDATION",
    "M(ox)": "OXIDATION",
    "S[167]": "PHOSPHO",
    "S[PHOSPHO]": "PHOSPHO",
    "S[PHOS]": "PHOSPHO",
    "T[181]": "PHOSPHO",
    "T[PHOSPHO]": "PHOSPHO",
    "T[PHOS]": "PHOSPHO",
    "Y[243]": "PHOSPHO",
    "K[Acetyl]": "ACETYL",
    "K[GlyGly]": "GG",
    "K[142]": "METHYL",
    "K[156]": "FORMYL",  # or "DIMETHYL",
    "P[113]": "OXIDATION",
    "R[170]": "METHYL",
    "R[184]": "DIMETHYL",
    "Y[208]": "NITRO",
}

# n[43] # TODO deal with terminal acetylation

MOD_AA_MASSES = AMINO_ACID.copy()
MOD_AA_MASSES.update(
    {
        k: AMINO_ACID[k[0]] + MODIFICATION.get(v, 0)
        for k, v in MOD_PEPTIDE_ALIASES.items()
    }
)

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
    "a": N_TERMINUS - CHO,
    "b": N_TERMINUS - H,
    "c": N_TERMINUS + NH2,
    "x": C_TERMINUS + CO - H,
    "y": C_TERMINUS + H,
    "z": C_TERMINUS - NH2,
}


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

del charge
del pos
del ion
del key

if __name__ == "__main__":
    my_vars = {k: v for k, v in globals().items() if not k.startswith("_")}
    for k, v in my_vars.items():
        print(f"\n>>> {k} {type(v)} = {v}")
