from transprosit import encoding_decoding
from transprosit import constants


def test_aa_encoding():
    out = encoding_decoding.encode_mod_seq("_AACD_")
    assert out[:10] == [1, 1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert len(out) == constants.MAX_SEQUENCE
