import warnings
from typing import List
from pathlib import Path

from elfragmentador import constants
from elfragmentador import annotate
from elfragmentador import encoding_decoding
from elfragmentador.encoding_decoding import get_fragment_encoding_labels

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


class Spectrum:
    def __init__(
        self,
        sequence: str,
        charge: int,
        parent_mz: float,
        mzs: List[float],
        intensities: List[float],
        modifications=None,
        tolerance=25,
        tolerance_unit="ppm",
        nce=None,
        instrument=None,
        analyzer=None,
        fragmentation=None,
        rt=None,
        raw_spectra=None,
    ):

        parsed_peptide = list(annotate.peptide_parser(sequence))
        # Makes sure all elements in the sequence are aminoacids
        assert set(parsed_peptide) <= constants.AMINO_ACID_SET.union(
            constants.MOD_PEPTIDE_ALIASES
        ), print(sequence)
        self.sequence = "".join([x[:1] for x in parsed_peptide])
        self.mod_sequence = sequence
        self.length = len(parsed_peptide)
        self.charge = charge
        self.parent_mz = parent_mz
        self.modifications = modifications

        amino_acids = annotate.peptide_parser(sequence)
        self.theoretical_mass = sum([constants.MOD_AA_MASSES[a] for a in amino_acids])
        self.theoretical_mz = (
            self.theoretical_mass + (charge * constants.PROTON)
        ) / charge

        assert len(mzs) == len(intensities)
        self.mzs = mzs
        self.intensities = intensities

        self.delta_m = abs(self.parent_mz - self.theoretical_mz)
        self.delta_ppm = 1e6 * abs(self.delta_m) / self.theoretical_mz

        # Annotation related section
        self.tolerance = tolerance
        self.tolerance_unit = tolerance_unit

        self._theoretical_peaks = annotate.get_peptide_ions(self.sequence)

        self._annotated_peaks = None
        self.nce = nce
        self.instrument = instrument
        self.analyzer = analyzer
        self.fragmentation = fragmentation
        self.rt = rt
        self.raw_spectra = raw_spectra

    @staticmethod
    def from_tensors(
        sequence_tensor, fragment_tensor, mod_tensor=None, *args, **kwargs
    ):
        mod_sequence = encoding_decoding.decode_mod_seq(sequence_tensor, mod_tensor)
        fragment_df = encoding_decoding.decode_fragment_tensor(
            mod_sequence, fragment_tensor
        )
        spec_out = Spectrum(
            mod_sequence,
            mzs=fragment_df["Mass"],
            intensities=fragment_df["Intensity"],
            *args,
            **kwargs,
        )
        return spec_out

    def precursor_error(self, error_type="ppm"):
        if error_type == "ppm":
            return self.delta_ppm
        elif error_type == "da":
            return self.delta_m
        else:
            raise NotImplementedError(
                "Not a know error type, select either of 'ppm' or 'da'"
            )

    def annotate_peaks(self) -> None:
        def in_tol(tmz, omz):
            out = annotate.is_in_tolerance(
                tmz, omz, tolerance=self.tolerance, unit=self.tolerance_unit
            )
            return out

        annots = {}
        for mz, inten in zip(self.mzs, self.intensities):
            matching = {
                k: inten for k, v in self._theoretical_peaks.items() if in_tol(v, mz)
            }
            annots.update(matching)

        max_int = max([v for v in annots.values()] + [0])
        annots = {k: v / max_int for k, v in annots.items()}
        self._annotated_peaks = annots

    def encode_spectra(self, dry=False):
        if self._annotated_peaks is None and not dry:
            self.annotate_peaks()

        if dry:
            peak_annot = None
        else:
            peak_annot = self._annotated_peaks

        return get_fragment_encoding_labels(annotated_peaks=peak_annot)

    def encode_sequence(self):
        return encoding_decoding.encode_mod_seq(self.mod_sequence)

    @property
    def annotated_peaks(self):
        if self._annotated_peaks is None:
            self._annotated_peaks

        return self.annotate_peaks()

    def __repr__(self) -> str:
        out = (
            "Spectrum:\n"
            f"\tSequence: {self.sequence} len:{self.length}\n"
            f"\tMod.Sequence: {self.mod_sequence}\n"
            f"\tCharge: {self.charge}\n"
            f"\tMZs: {self.mzs[:3]}...{len(self.mzs)}\n"
            f"\tInts: {self.intensities[:3]}...{len(self.intensities)}\n"
            f"\tInstrument: {self.fragmentation}\n"
            f"\tInstrument: {self.instrument}\n"
            f"\tAnalyzer: {self.analyzer}\n"
            f"\tNCE: {self.nce}\n"
            f"\tRT: {self.rt}\n"
            f"\tOriginalSpectra: {self.raw_spectra}\n"
        )

        if self._annotated_peaks is not None:
            out += f"\tAnnotations: {self._annotated_peaks}\n"

        return out


def encode_sptxt(filepath, max_spec=1e9, irt_fun=None, *args, **kwargs):
    iter = read_sptxt(filepath, *args, **kwargs)

    sequences = []
    mod_sequences = []
    seq_encodings = []
    mod_encodings = []
    spectra_encodings = []
    charges = []
    rts = []
    nces = []
    orig = []

    for i, spec in enumerate(tqdm(iter)):
        if i >= max_spec:
            break
        seq_encode, mod_encode = spec.encode_sequence()
        seq_encode, mod_encode = str(seq_encode), str(mod_encode)

        spectra_encodings.append(str(spec.encode_spectra()))
        seq_encodings.append(seq_encode)
        mod_encodings.append(mod_encode)
        charges.append(spec.charge)
        sequences.append(spec.sequence)
        mod_sequences.append(spec.mod_sequence)
        rts.append(spec.rt)
        nces.append(spec.nce)
        orig.append(spec.raw_spectra)

    ret = pd.DataFrame(
        {
            "Sequences": sequences,
            "ModSequences": mod_sequences,
            "SpectraEncodings": spectra_encodings,
            "ModEncodings": mod_encodings,
            "SeqEncodings": seq_encodings,
            "Charges": charges,
            "RTs": rts,
            "NCEs": nces,
            "OrigSpectra": orig,
        }
    )

    if irt_fun is not None:
        raise NotImplementedError
    else:
        warnings.warn(
            "No calculation function passed for iRT,"
            " will replace the column with missing"
        )
        ret["iRT"] = np.nan

    return ret


def read_sptxt(filepath: Path, *args, **kwargs) -> List[Spectrum]:
    """
    read_sptxt reads a spectra library file

    reads a spectral library file into a list of spectra objects

    Parameters
    ----------
    filepath : Path
        The path to the spectral library, extension .sptxt
    *args
        Passed onto Spectrum
    **kwargs
        Passed onto Spectrum

    Yields
    -------
    Spectrum objects
    """
    with open(filepath, "r") as f:
        spectrum_section = []
        for line in f:
            if line.startswith("#"):
                continue
            stripped_line = line.strip()
            if len(stripped_line) == 0:
                if len(spectrum_section) > 0:
                    yield _parse_spectra_sptxt(spectrum_section, *args, **kwargs)
                    spectrum_section = []
            else:
                spectrum_section.append(stripped_line)

        if len(spectrum_section) > 0:
            yield _parse_spectra_sptxt(spectrum_section, *args, **kwargs)


def _parse_spectra_sptxt(x, instrument=None, analyzer=None, *args, **kwargs):
    """
    Parses a single spectra into an object

    Meant for internal use
    """
    digits = [str(v) for v in range(10)]

    # Header Handling
    named_params = [v for v in x if ":" in v]
    named_params_dict = {}
    for v in named_params:
        tmp = v.split(":")
        named_params_dict[tmp[0].strip()] = tmp[1]

    fragmentation = named_params_dict.get("FullName", None)
    if fragmentation is not None:
        fragmentation = fragmentation[fragmentation.index("(") + 1 : -1]

    comment_sec = [v.split("=") for v in named_params_dict["Comment"].strip().split()]
    comment_dict = {v[0]: v[1] for v in comment_sec}
    sequence, charge = named_params_dict["Name"].split("/")

    nce = comment_dict.get("CollisionEnergy", None)
    if nce is not None:
        nce = float(nce)

    rt = comment_dict.get("RetentionTime", None)
    if rt is not None:
        rt = float(rt.split(",")[0])

    raw_spectra = comment_dict.get("RawSpectrum", None)

    # Peaks Handling
    peaks_sec = [v for v in x if v[0] in digits and ("\t" in v or " " in v)]
    peaks_sec = [l.strip().split() for l in peaks_sec if "." in l]
    mz = [float(l[0]) for l in peaks_sec]
    intensity = [float(l[1]) for l in peaks_sec]

    out_spec = Spectrum(
        sequence=sequence.strip(),
        charge=int(charge),
        parent_mz=float(comment_dict["Parent"]),
        intensities=intensity,
        mzs=mz,
        modifications=comment_dict["Mods"],
        fragmentation=fragmentation,
        analyzer=analyzer,
        instrument=instrument,
        nce=nce,
        rt=rt,
        raw_spectra=raw_spectra,
        *args,
        **kwargs,
    )

    return out_spec
