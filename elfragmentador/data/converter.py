import torch
from ms2ml import AnnotatedPeptideSpectrum, Peptide

from elfragmentador.config import CONFIG
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

    CONFIG = CONFIG

    @classmethod
    def convert_annotated_spectrum(
        cls, data: AnnotatedPeptideSpectrum, nce: float
    ) -> TrainBatch:
        out = TrainBatch(
            seq=torch.from_numpy(data.precursor_peptide.aa_to_vector()).unsqueeze(0),
            mods=torch.from_numpy(data.precursor_peptide.mod_to_vector()).unsqueeze(0),
            charge=torch.Tensor([data.precursor_peptide.charge]).unsqueeze(0),
            nce=torch.Tensor([nce]).unsqueeze(0),
            spectra=torch.from_numpy(data.encode_fragments()).unsqueeze(0),
            irt=torch.Tensor([data.retention_time.minutes() / 100]).unsqueeze(0),
            weight=torch.Tensor([1]).unsqueeze(0),
        )
        return out

    @classmethod
    def convert_peptide(cls, data: Peptide, nce: float) -> ForwardBatch:
        out = ForwardBatch(
            seq=torch.from_numpy(data.aa_to_vector()).unsqueeze(0),
            mods=torch.from_numpy(data.mod_to_vector()).unsqueeze(0),
            charge=torch.Tensor([data.charge]).unsqueeze(0),
            nce=torch.Tensor([nce]).unsqueeze(0),
        )

        return out

    @classmethod
    def convert_string(cls, data: str, nce: float) -> ForwardBatch:
        return cls.convert_peptide(
            data=Peptide.from_proforma_seq(data, config=cls.CONFIG), nce=nce
        )


class DeTensorizer:
    """Converts tensors to spectra or peptides"""

    CONFIG = CONFIG

    @classmethod
    def make_spectrum(cls, seq, mod, charge, fragment_vector):
        spec = AnnotatedPeptideSpectrum.decode_fragments(
            fragment_vector=fragment_vector,
            peptide=cls.make_peptide(seq=seq, mod=mod, charge=charge),
        )
        return spec

    @classmethod
    def make_peptide(cls, seq, mod, charge: int | None = None):
        peptide = Peptide.decode_vector(
            config=cls.CONFIG, seq=seq, mod=mod, charge=int(charge)
        )
        return peptide
