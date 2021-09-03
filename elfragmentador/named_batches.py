
from collections import namedtuple

import torch
from torch import Tensor

PredictionResults = namedtuple("PredictionResults", "irt spectra")
PredictionResults.__doc__ = """Named Tuple that bundles prediction results
Parameters:
    irt (Tensor): Tensor containing normalized irt predictions
    spectra (Tensor): Tensor containing encoded predicted spectra

Examples:
    >>> PredictionResults(irt = torch.rand(43), spectra = torch.rand(1))
    PredictionResults(irt=tensor([...]), spectra=tensor([0...]))
"""

ForwardBatch = namedtuple("ForwardBatch", "seq mods charge nce")
ForwardBatch.__doc__ = """Named Tuple that bundles all tensors needed for a forward pass in the model
Parameters:
    seq (Tensor): Encoded peptide sequences
    mods (Tensor): Modification encodings for the sequence
    charge (Tensor): Long tensor with the charges
    nce (Tensor): Normalized collision energy

Examples:
    >>> ForwardBatch(
    ... seq = torch.ones(12),
    ... mods=torch.zeros(12),
    ... charge=torch.ones(12) * 2,
    ... nce = torch.ones(12) * 34)
    ForwardBatch(seq=tensor([1., ...]), mods=tensor([0., ...]), charge=tensor([2., ...]), nce=tensor([34., ...]))
"""

TrainBatch = namedtuple(
    "TrainBatch",
    "seq, mods, charge, nce, spectra, irt, weight",
)
TrainBatch.__doc__ = """Named Tuple that bundles all tensors needed for a training step in the model
Parameters:
    seq (Tensor): Encoded peptide sequences
    mods (Tensor): Modification encodings for the sequence
    charge (Tensor): Long tensor with the charges
    nce (Tensor): Normalized collision energy
    irt (Tensor): Tensor containing normalized irt predictions
    spectra (Tensor): Tensor containing encoded predicted spectra
    weight (Union[Tensor, None]): Weight of the element for the loss

Examples:
    >>> x = ForwardBatch(
    ... seq = torch.ones(12),
    ... mods=torch.zeros(12),
    ... charge=torch.ones(12) * 2,
    ... nce = torch.ones(12) * 34)
    >>> y = PredictionResults(spectra = torch.rand(43), irt = torch.rand(1))
    >>> TrainBatch(**x._asdict(), **y._asdict(), weight=None)
    TrainBatch(seq=tensor([...]), mods=tensor([...]), charge=tensor([...]), nce=tensor([...]), spectra=tensor([...]), irt=tensor([...]), weight=None)
"""