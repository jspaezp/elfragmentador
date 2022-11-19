from __future__ import annotations

import torch
from loguru import logger
from ms2ml import AnnotatedPeptideSpectrum, Peptide
from ms2ml.annotation_classes import RetentionTime

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
        >>> converter(spec, nce=27) # doctest: +ELLIPSIS
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
        irt = RTConverter.to_tensor(data.retention_time)
        out = TrainBatch(
            seq=torch.from_numpy(data.precursor_peptide.aa_to_vector()).unsqueeze(0),
            mods=torch.from_numpy(data.precursor_peptide.mod_to_vector()).unsqueeze(0),
            charge=torch.Tensor([data.precursor_peptide.charge]).unsqueeze(0),
            nce=torch.Tensor([nce]).unsqueeze(0),
            spectra=torch.from_numpy(data.encode_fragments()).unsqueeze(0),
            irt=irt,
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

    def __call__(self, data, nce):
        if isinstance(data, Peptide):
            return self.convert_peptide(data, nce)
        if isinstance(data, AnnotatedPeptideSpectrum):
            return self.convert_annotated_spectrum(data, nce)


class RTConverter:
    warned_once = False

    @classmethod
    def to_tensor(cls, rt: RetentionTime | float) -> torch.Tensor:
        try:
            irt = torch.Tensor([rt.minutes() / 100]).unsqueeze(0)
        except AttributeError:
            if not cls.warned_once:
                logger.warning(
                    "Retention time was not correctly encoded in the data, seconds"
                    "and minutes might be mixed up. Please report the issue"
                )
                cls.warned_once = True
            irt = torch.Tensor([rt]).unsqueeze(0)
        return irt

    @classmethod
    def to_seconds(cls, rt: RetentionTime | float) -> float:
        try:
            return rt.minutes() * 60
        except AttributeError:
            if not cls.warned_once:
                logger.warning(
                    "Retention time was not correctly encoded in the data, seconds"
                    "and minutes might be mixed up. Please report the issue"
                )
                cls.warned_once = True
        return float(rt)


class DeTensorizer:
    """Converts tensors to spectra or peptides"""

    CONFIG = CONFIG
    warned_once_rt = False

    @classmethod
    def make_spectrum(cls, seq, mod, charge, fragment_vector, irt):
        fragment_vector = torch.relu(fragment_vector) / fragment_vector.max()
        fragment_vector = fragment_vector.clone().detach().cpu().numpy().squeeze()
        spec = AnnotatedPeptideSpectrum.decode_fragments(
            fragment_vector=fragment_vector,
            peptide=cls.make_peptide(seq=seq, mod=mod, charge=charge),
        )
        if irt > 2 and not cls.warned_once_rt:
            logger.warning(
                f"Passed Retention time {irt} is >2, this might be correct"
                " but could also mean that the rt is not scaled"
                " (as expected by the converter)."
                "RTs are expected to be biognosys-irt/100"
            )
            cls.warned_once_rt = True
        spec.retention_time = RetentionTime(float(irt) * 100 * 60, "s")
        return spec

    @classmethod
    def make_peptide(cls, seq, mod, charge: int | None = None):
        seq = seq.squeeze().clone().detach().cpu().numpy()
        mods = mod.squeeze().clone().detach().cpu().numpy()
        charge = charge.squeeze().clone().detach().cpu().numpy()

        peptide = Peptide.decode_vector(
            config=cls.CONFIG, seq=seq, mod=mods, charge=int(charge)
        )
        return peptide
