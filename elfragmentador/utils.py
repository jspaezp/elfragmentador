from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.utils.data.dataset import TensorDataset

import elfragmentador.constants as CONSTANTS
from elfragmentador import encoding_decoding
from elfragmentador.named_batches import ForwardBatch

# TODO split addition of metadata and actual predictions to separate functions to


def torch_batch_from_seq(
    seq: str, nce: float, charge: int, enforce_length=True, pad_zeros=True
):
    """
    Generate an input batch for the model from a sequence string.

    Parameters:
        seq (str):
            String describing the sequence to be predicted, e. "PEPT[PHOSOHO]IDEPINK"
        nce (float): Collision energy to use for the prediction, e. 27.0
        charge (int): Charge of the precursor to use for the prediction, e. 3

    Returns:
        ForwardBatch: Named tuple with the tensors to use as a forward batch

    Examples:
        >>> torch_batch_from_seq("PEPTIDEPINK", 27.0, 3)
        ForwardBatch(seq=tensor([[23, 13,  4, 13, 17,  ...]]), \
            mods=tensor([[0, ... 0]]), \
            charge=tensor([[3]]), nce=tensor([[27.]]))
    """
    encoded_seq, encoded_mods = encoding_decoding.encode_mod_seq(
        seq, enforce_length=enforce_length, pad_zeros=pad_zeros
    )

    seq = torch.Tensor(encoded_seq).unsqueeze(0).long()
    mods = torch.Tensor(encoded_mods).unsqueeze(0).long()
    in_charge = torch.Tensor([[charge]]).long()
    in_nce = torch.Tensor([[nce]]).float()

    # This is a named tuple
    out = ForwardBatch(seq=seq, mods=mods, nce=in_nce, charge=in_charge)
    return out


def _attempt_find_file(row_rawfile, possible_paths):
    tried_paths = []
    for pp in possible_paths:
        rawfile_path = Path(pp) / (row_rawfile + ".mzML")
        tried_paths.append(rawfile_path)

        if rawfile_path.is_file():
            return rawfile_path
        else:
            logging.debug(f"{rawfile_path}, not found")

    logging.error(f"File not found in any of: {[str(x) for x in tried_paths]}")
    raise FileNotFoundError(tried_paths)


def get_random_peptide():
    AAS = [x for x in CONSTANTS.ALPHABET if x.isupper()]
    len_pep = random.randint(5, CONSTANTS.MAX_SEQUENCE)
    out_pep = ""

    for _ in range(len_pep):
        out_pep += "".join(random.sample(AAS, 1))

    return out_pep


def _concat_batches(batches):
    out = []
    for i, _ in enumerate(batches[0]):
        out.append(torch.cat([b[i] for b in batches]))

    return tuple(out)


def prepare_fake_tensor_dataset(num=50):
    peps = [
        {
            "nce": 20 + (10 * random.random()),
            "charge": random.randint(1, 5),
            "seq": get_random_peptide(),
        }
        for _ in range(num)
    ]

    tensors = [torch_batch_from_seq(**pep) for pep in peps]
    tensors = TensorDataset(*_concat_batches(batches=tensors))

    return tensors
