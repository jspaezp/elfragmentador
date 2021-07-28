import torch
from elfragmentador.nn_encoding import SeqPositionalEmbed


def test_inverted_positional_encoder():
    encoder = SeqPositionalEmbed(6, 50, inverted=True)
    x = torch.cat(
        [torch.ones(1, 2), torch.ones(1, 2) * 2, torch.zeros((1, 2))], dim=-1
    ).long()
    x[0]
    # tensor([1, 1, 2, 2, 0, 0])
    x.shape
    # torch.Size([1, 6])
    out = encoder(x)
    assert out.shape == torch.Size((6, 1, 6))

    # Check that the embedding of the empty encodings are empty
    assert torch.all(out[5, 0, :] == 0)
    assert torch.all(out[4, 0, :] == 0)

    # Check that the embedding of a non empty is not empty
    assert torch.all(out[3, 0, :] != 0)
    assert torch.all(out[3, 0, :] == encoder.pe[1])
    assert torch.any(out[2, 0, :] == encoder.pe[2])
    assert torch.any(out[1, 0, :] == encoder.pe[3])
    assert torch.any(out[0, 0, :] == encoder.pe[4])
    assert torch.any(out[3, 0, :] != out[0, 0, :])

    encoder = SeqPositionalEmbed(6, 50, inverted=True)
    input_t = torch.tril(torch.ones((3, 3))).long()
    input_t[1]
    # tensor([1, 1, 0])
    out = encoder(input_t)
    assert out.shape == torch.Size((3, 3, 6))
    assert torch.all(out[2, 1, :] == 0)
    assert torch.any(out[1, 1, :] != 0)
    assert torch.any(out[0, 1, :] != 0)
    assert torch.any(out[0, 1, :] != out[1, 1, :])
