import torch
from transprosit import encoding_decoding
from transprosit import constants


def test_aa_encoding():
    out = encoding_decoding.encode_mod_seq("_AACD_")
    assert out[:10] == [1, 1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert len(out) == constants.MAX_SEQUENCE


def test_fragment_encoding_decoding():
    encoding_decoding.get_fragment_encoding_labels()
    encoding_decoding.get_fragment_encoding_labels({"z1y2": 100, "z2y2": 52})
    encoding_decoding.decode_fragment_tensor(
        "AAACK", torch.rand((constants.NUM_FRAG_EMBEDINGS))
    )
