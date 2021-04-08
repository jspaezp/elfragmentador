"""
Provides functions to allow experimentation and ablations on the model and data.
"""

import warnings
from argparse import ArgumentParser
from elfragmentador.model import PepTransformerModel


def add_experimental_parser_options(parser: ArgumentParser):
    parser.add_argument("--ablate_rts", type=bool, default=False)
    parser.add_argument("--ablate_nce", type=bool, default=False)
    parser.add_argument("--ablate_pos_encoding", type=bool, default=False)


def ablate_rts(dataloader):
    raise NotImplementedError


def test_ablate_rts():
    raise NotImplementedError


def ablate_nce(dataloader):
    raise NotImplementedError


def test_ablate_nce(dataloader):
    raise NotImplementedError


def ablate_positional_encoding(model: PepTransformerModel):
    warnings.warn(
        50 * "\n>>>> Dropping positional encodings, make sure you want that <<<<\n"
    )
    model.encoder.pos_encoder.pe[:] = 0
    return model
    raise NotImplementedError
