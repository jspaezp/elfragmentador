"""
PredictionResults = irt + spec
ForwardBatch = seq + mods + charge + nce
TrainBatch = PredictionResults + ForwardBatch + Weight

EvaluationLossBatch = scaled_se_loss + loss_cosine + loss_irt + loss_angle
EvaluationPredictionBatch = EvaluationLossBatch + PredictionBatch
"""

from typing import NamedTuple, Union

from torch import Tensor


def _hash_tensors(tensor_tuple):
    return sum(
        [
            hash(x.data.tolist().__str__()) if isinstance(x, Tensor) else hash(x)
            for x in tensor_tuple
        ]
    )


class PredictionResults(NamedTuple):
    """
    Named Tuple that bundles prediction results.

    Parameters:
        irt (Tensor): Tensor containing normalized irt predictions
        spectra (Tensor): Tensor containing encoded predicted spectra

    Examples:
        >>> import torch
        >>> foo = PredictionResults(irt = torch.rand(43), spectra = torch.rand(1))
        >>> foo
        PredictionResults(irt=tensor([...]), spectra=tensor([0...]))
        >>> foo2 = PredictionResults(*[x.clone() for x in foo])
        >>> hash(foo) == hash(foo2)
        True
    """

    irt: Tensor
    spectra: Tensor

    def __hash__(self):
        return _hash_tensors(self)


class ForwardBatch(NamedTuple):
    """
    Named Tuple that bundles all tensors needed for a forward pass in the
    model.

    Parameters:
        seq (Tensor): Encoded peptide sequences
        mods (Tensor): Modification encodings for the sequence
        charge (Tensor): Long tensor with the charges
        nce (Tensor): Normalized collision energy

    Examples:
        >>> import torch
        >>> ForwardBatch(
        ... seq = torch.ones(2),
        ... mods=torch.zeros(2),
        ... charge=torch.ones(2) * 2,
        ... nce = torch.ones(2) * 34)
        ForwardBatch(seq=tensor([1., 1.]), mods=tensor([0., 0.]), \
charge=tensor([2., 2.]), nce=tensor([34., 34.]))
    """

    seq: Tensor
    mods: Tensor
    charge: Tensor
    nce: Tensor

    def __hash__(self):
        return _hash_tensors(self)


class TrainBatch(NamedTuple):
    """
    Named Tuple that bundles all tensors needed for a training step in the
    model.

    Parameters:
        seq (Tensor): Encoded peptide sequences
        mods (Tensor): Modification encodings for the sequence
        charge (Tensor): Long tensor with the charges
        nce (Tensor): Normalized collision energy
        irt (Tensor): Tensor containing normalized irt predictions
        spectra (Tensor): Tensor containing encoded predicted spectra
        weight (Union[Tensor, None]): Weight of the element for the loss

    Examples:
        >>> import torch
        >>> x = ForwardBatch(
        ... seq = torch.ones(12),
        ... mods=torch.zeros(12),
        ... charge=torch.ones(12) * 2,
        ... nce = torch.ones(12) * 34)
        >>> y = PredictionResults(spectra = torch.rand(43), irt = torch.rand(1))
        >>> TrainBatch(**x._asdict(), **y._asdict(), weight=None)
        TrainBatch(seq=tensor([...]), mods=tensor([...]), charge=tensor([...]), \
            nce=tensor([...]), spectra=tensor([...]), irt=tensor([...]), weight=None)
    """

    seq: Tensor
    mods: Tensor
    charge: Tensor
    nce: Tensor
    spectra: Tensor
    irt: Tensor
    weight: Tensor

    def __hash__(self):
        return _hash_tensors(self)


class EvaluationLossBatch(NamedTuple):
    """Named tuple that bundles losses from evaluating against a dataset
    Parameters:
        scaled_se_loss (Tensor):
            Squared error of the scaled versions of the input
            and target retention times.
            (mean subtraction and divided by standard deviation)
        loss_cosine (Tensor):
            1 - Cosine similarity of the spectra
        loss_irt (Tensor):
            Squared error of the non-scaled retention times
        loss_angle (Tensor):
            1 - Spectral angle loss

    """

    scaled_se_loss: Tensor
    loss_cosine: Tensor
    loss_irt: Tensor
    loss_angle: Tensor

    def __hash__(self):
        return _hash_tensors(self)


class EvaluationPredictionBatch(NamedTuple):
    """
    "EvaluationPredictionBatch.

    Parameters:
        scaled_se_loss (Tensor):
            Squared error of the scaled versions of the input
            and target retention times.
            (mean subtraction and divided by standard deviation)
        loss_cosine (Tensor):
            1 - Cosine similarity of the spectra
        loss_irt (Tensor):
            Squared error of the non-scaled retention times
        loss_angle (Tensor):
            1 - Spectral angle loss
        irt (Tensor):
            Tensor containing normalized irt predictions
        spectra (Tensor):
            Tensor containing encoded predicted spectra
    """

    scaled_se_loss: Tensor
    loss_cosine: Tensor
    loss_irt: Tensor
    loss_angle: Tensor
    irt: Tensor
    spectra: Tensor

    def __hash__(self):
        return _hash_tensors(self)


NamedTensorBatch = Union[
    EvaluationLossBatch,
    EvaluationPredictionBatch,
    ForwardBatch,
    TrainBatch,
    PredictionResults,
]
