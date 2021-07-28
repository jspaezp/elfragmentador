import pytest

import torch
from elfragmentador import encoding_decoding
from elfragmentador import constants


testdata_aa_encoding = [
    ("_AACD_", [23, 1, 1, 2, 3, 22, 0, 0, 0, 0, 0], "AACD"),
    ("n[43]AACD", [23, 1, 1, 2, 3, 22, 0, 0, 0, 0, 0], "n[ACETYL]AACD"),
    ("AAAC[160]DDDK", None, "AAACDDDK"),
    ("M[147]AAAK", None, "M[OXIDATION]AAAK"),
    ("AAACSS[167]", None, "AAACSS[PHOSPHO]"),
    ("KAKT[181]AA", None, "KAKT[PHOSPHO]AA"),
    ("KAKY[243]FG", None, "KAKY[PHOSPHO]FG"),
    ("KAKY[+80]FG", None, "KAKY[PHOSPHO]FG"),
]


@pytest.mark.parametrize(
    "input_sequence,expected_first_10_encoding,expected_output_sequence",
    testdata_aa_encoding,
)
def test_aa_encoding(
    input_sequence, expected_first_10_encoding, expected_output_sequence
):
    out, mods_out = encoding_decoding.encode_mod_seq(input_sequence)

    if expected_first_10_encoding is not None:
        assert out[:10] == expected_first_10_encoding[:10]
    assert len(out) == constants.MAX_TENSOR_SEQUENCE
    decoded = encoding_decoding.decode_mod_seq(out, mods_out)
    assert decoded == expected_output_sequence


def test_fragment_encoding_decoding():
    encoding_decoding.get_fragment_encoding_labels()
    encoding_decoding.get_fragment_encoding_labels({"z1y2": 100, "z2y2": 52})
    encoding_decoding.decode_fragment_tensor(
        "AAACK", torch.rand((constants.NUM_FRAG_EMBEDINGS))
    )
