"""
Greatly inspired/copied from:
https://github.com/kusterlab/prosit/blob/master/prosit/constants.py

And released under an Apache 2.0 license
"""

VAL_SPLIT = 0.8

TOLERANCE_FTMS = 25
TOLERANCE_ITMS = 0.35
TOLERANCE_TRIPLETOF = 0.5

TOLERANCE = {"FTMS": (25, "ppm"), "ITMS": (0.35, "da"), "TripleTOF": (50, "ppm")}

ALPHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
}
ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
AAS_NUM = len(ALPHABET)

CHARGES = [1, 2, 3, 4, 5, 6]
DEFAULT_MAX_CHARGE = max(CHARGES)
MAX_FRAG_CHARGE = 3
MAX_SEQUENCE = 30
MAX_ION = MAX_SEQUENCE - 1
ION_TYPES = ["y", "b"]
sorted(ION_TYPES)
NLOSSES = ["", "H2O", "NH3"]

FORWARD = {"a", "b", "c"}
BACKWARD = {"x", "y", "z"}

# Amino acids

# Modifications use high caps PSI-MS name

MODIFICATION = {
    "CARBAMIDOMETHYL": 57.0214637236,  # Carbamidomethylation (CAM)
    "OXIDATION": 15.99491,  # Oxidation
    "PHOSPHO": 79.966331,  # Phosphorylation
    "METHYL": 14.015650,  # Methylation
    "DIMETHYL": 28.031300,  # Dimethylation
}

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
