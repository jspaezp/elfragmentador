import torch
from elfragmentador import encoding_decoding
from elfragmentador import constants


def test_aa_encoding():
    out, _ = encoding_decoding.encode_mod_seq("_AACD_")
    assert out[:10] == [1, 1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert len(out) == constants.MAX_SEQUENCE
    decoded = encoding_decoding.decode_mod_seq(out)
    assert decoded == "AACD"


def test_mod_aa_encoding():
    test_seqs = [
        "AAAC[160]DDDK",
        "M[147]AAAK",
        "AAACSS[167]",
        "KAKT[181]AA",
        "KAKY[243]FG",
    ]

    for s in test_seqs:
        out_s, out_m = encoding_decoding.encode_mod_seq(s)
        assert len(out_s) == constants.MAX_SEQUENCE
        decoded = encoding_decoding.decode_mod_seq(out_s, out_m)
        s = s.replace("C[160]", "C")
        for k, v in constants.MOD_PEPTIDE_ALIASES.items():
            s = s.replace(k, f"{k[0]}[{v}]")

        assert s == decoded


def test_fragment_encoding_decoding():
    encoding_decoding.get_fragment_encoding_labels()
    encoding_decoding.get_fragment_encoding_labels({"z1y2": 100, "z2y2": 52})
    encoding_decoding.decode_fragment_tensor(
        "AAACK", torch.rand((constants.NUM_FRAG_EMBEDINGS))
    )
