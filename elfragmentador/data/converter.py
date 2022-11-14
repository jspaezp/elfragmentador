from functools import singledispatchmethod
from typing import Any

import torch
from loguru import logger as lg_logger
from ms2ml import AnnotatedPeptideSpectrum, Peptide

from elfragmentador.config import get_default_config
from elfragmentador.named_batches import ForwardBatch, TrainBatch


class Tensorizer:
    """Converts peptides and spectra to tensors.


    Example:
        >>> from elfragmentador.data.converter import Tensorizer
        >>> from elfragmentador.config import get_default_config
        >>> from ms2ml import AnnotatedPeptideSpectrum
        >>> converter = Tensorizer()
        >>> spec = AnnotatedPeptideSpectrum._sample()
        >>> spec.config = get_default_config()
        >>> converter(spec) # doctest: +ELLIPSIS
        TrainBatch(seq=tensor([[ 0, 16,  5, 16, 16,  9, 14, 11, 27]]),
            mods=tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            charge=tensor([[2.]]),
            nce=tensor([[27.]]),
            spectra=tensor([[...]]),
            irt=tensor([[nan]]),
            weight=tensor([[1.]]))

    """

    def __init__(self, nce=None) -> None:
        if nce is None:
            lg_logger.warning("No NCE value provided, using 27.0 as a default")
            nce = 27.0

        self.nce = nce

    @singledispatchmethod
    def convert(self, data):
        raise NotImplementedError

    @convert.register(AnnotatedPeptideSpectrum)
    def _(self, data: AnnotatedPeptideSpectrum) -> TrainBatch:
        out = TrainBatch(
            seq=torch.from_numpy(data.precursor_peptide.aa_to_vector()).unsqueeze(0),
            mods=torch.from_numpy(data.precursor_peptide.mod_to_vector()).unsqueeze(0),
            charge=torch.Tensor([data.precursor_peptide.charge]).unsqueeze(0),
            nce=torch.Tensor([self.nce]).unsqueeze(0),
            spectra=torch.from_numpy(data.encode_fragments()).unsqueeze(0),
            irt=torch.Tensor([data.retention_time.minutes() / 100]).unsqueeze(0),
            weight=torch.Tensor([1]).unsqueeze(0),
        )
        return out

    @convert.register(Peptide)
    def _(self, data: Peptide):
        out = ForwardBatch(
            seq=torch.from_numpy(data.aa_to_vector()).unsqueeze(0),
            mods=torch.from_numpy(data.mod_to_vector()).unsqueeze(0),
            charge=torch.Tensor([data.charge]).unsqueeze(0),
            nce=torch.Tensor([self.nce]).unsqueeze(0),
        )

        return out

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.convert(*args, **kwds)


class DeTensorizer:
    """Converts tensors to spectra or peptides"""

    CONFIG = get_default_config()

    def make_spectrum(self, seq, mod, charge, fragment_vector):
        spec = AnnotatedPeptideSpectrum.decode_fragments(
            fragment_vector=fragment_vector,
            peptide=self.make_peptide(seq=seq, mod=mod, charge=charge),
        )
        return spec

    def make_peptide(self, seq, mod, charge: int | None = None):
        peptide = Peptide.decode_vector(
            config=self.CONFIG, seq=seq, mod=mod, charge=int(charge)
        )
        return peptide
