from __future__ import annotations

import torch
from ms2ml import Peptide

from elfragmentador.config import get_default_config
from elfragmentador.named_batches import ForwardBatch

DEFAULT_CONFIG = get_default_config()


def torch_batch_from_seq(
    seq: str,
    nce: float,
    charge: int,
):
    """
    Generate an input batch for the model from a sequence string.

    Note that it is intented to em,ulate a batch of size 1, so the output
    not a single element.

    Parameters:
        seq (str):
            String describing the sequence to be predicted, e. "PEPT[PHOSOHO]IDEPINK"
        nce (float): Collision energy to use for the prediction, e. 27.0
        charge (int): Charge of the precursor to use for the prediction, e. 3

    Returns:
        ForwardBatch: Named tuple with the tensors to use as a forward batch

    Examples:
        >>> torch_batch_from_seq("PEPTIDEPINK", 27.0, 3)
        ForwardBatch(seq=tensor([[ 0, 16,  5, 16, 20,  9,  4,  5, 16,  9, 14, 11, 27]]),
            mods=tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            charge=tensor([[3]]),
            nce=tensor([[27.]]))
    """
    pep = Peptide.from_proforma_seq(seq, config=DEFAULT_CONFIG)
    encoded_seq, encoded_mods = pep.aa_to_vector(), pep.mod_to_vector()

    seq = torch.Tensor(encoded_seq).unsqueeze(0).long()
    mods = torch.Tensor(encoded_mods).unsqueeze(0).long()
    in_charge = torch.Tensor([[charge]]).long()
    in_nce = torch.Tensor([[nce]]).float()

    # This is a named tuple
    out = ForwardBatch(seq=seq, mods=mods, nce=in_nce, charge=in_charge)
    return out


def _concat_batches(batches):
    """Concatenates batches

    The output is menat tob e used in a tensor dataset
    """
    out = []
    for i, _ in enumerate(batches[0]):
        out.append(torch.cat([b[i] for b in batches]))

    return tuple(out)
