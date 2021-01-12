from typing import List
from pathlib import Path

from transprosit import constants
from transprosit import annotate

import pandas as pd
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
        ion_types="by",
        tolerance=25,
        tolerance_unit="ppm",
    ):

        # Makes sure all elements in the sequence are aminoacids
        assert set(sequence) <= constants.AMINO_ACID_SET, print(sequence)
        self.sequence = sequence
        self.charge = charge
        self.parent_mz = parent_mz
        self.modifications = modifications

        amino_acids = annotate.peptide_parser(sequence)
        self.theoretical_mass = sum([constants.AMINO_ACID[a] for a in amino_acids])
        self.theoretical_mz = (
            self.theoretical_mass + (charge * constants.PROTON)
        ) / charge

        assert len(mzs) == len(intensities)
        self.mzs = mzs
        self.intensities = intensities

        self.delta_m = abs(self.parent_mz - self.theoretical_mz)
        self.delta_ppm = 1e6 * abs(self.delta_m) / self.theoretical_mz

        # Annotation related section
        self.ion_types = "".join(sorted(ion_types))
        self.tolerance = tolerance
        self.tolerance_unit = tolerance_unit

        self._theoretical_peaks = annotate.get_peptide_ions(
            self.sequence, charges=range(1, self.charge), ion_types="by"
        )

        self._annotated_peaks = None

    def precursor_error(self, error_type="ppm"):
        if error_type == "ppm":
            return self.delta_ppm
        elif error_type == "da":
            return self.delta_m
        else:
            raise NotImplementedError(
                "Not a know error type, select either of 'ppm' or 'da'"
            )

    def annotate_peaks(self):
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

    def encode_annotations(self, max_charge=3, ions="yb", max_length=25, dry=False):
        if self._annotated_peaks is None and not dry:
            self.annotate_peaks()

        if dry:
            peak_annot = None
        else:
            peak_annot = self._annotated_peaks

        return get_fragment_encoding_labels(
            max_charge=max_charge, ions=ions, max_length=max_length
        )

    @property
    def annotated_peaks(self):
        if self._annotated_peaks is None:
            self._annotated_peaks

        return self.annotate_peaks()

    def __repr__(self):
        out = (
            "Spectrum:\n"
            f"\tSequence: {self.sequence} len:{len(self.sequence)}\n"
            f"\tCharge: {self.charge}\n"
            f"\tMZs: {self.mzs[:3]}...{len(self.mzs)}\n"
            f"\tInts: {self.intensities[:3]}...{len(self.intensities)}\n"
        )

        if self._annotated_peaks is not None:
            out += f"\tAnnotations: {self._annotated_peaks}\n"

        return out


def decode_tensor(sequence, tensor, max_charge, ions, max_length):
    key_list = get_fragment_encoding_labels(
        max_charge=max_charge, ions=ions, max_lenght=max_length, dry=True
    )
    fragment_ions = annotate.get_peptide_ions(
        sequence, list(1, max_charge), ion_types=ions
    )
    masses = [fragment_ions[k] for k in key_list]
    intensities = [float(x) for x in tensor]

    return pd.DataFrame(
        {"Fragment": fragment_ions, "Mass": masses, "Intensity": intensities}
    )


def get_fragment_encoding_labels(
    max_charge=3, ions="yb", max_length=25, annotated_peaks=None
):
    ions = "".join(sorted(ions))
    charges = list(range(1, max_charge + 1))
    positions = list(range(1, max_length + 1))

    encoding = []
    # TODO implement neutral losses ...  if needed
    for pos in positions:
        for charge in charges:
            for ion in ions:
                key = f"z{charge}{ion}{pos}"
                if annotated_peaks is None:
                    encoding.append(key)
                else:
                    encoding.append(annotated_peaks.get(key, 0))

    return encoding


def encode_sptxt(filepath, *args, **kwargs):
    iter = read_sptxt(filepath, *args, **kwargs)

    encodings = []
    sequences = []
    charges = []
    for spec in tqdm(iter):
        encodings.append(str(spec.encode_annotations()))
        charges.append(spec.charge)
        sequences.append(spec.sequence)

    ret = pd.DataFrame(
        {
            "Sequences": sequences,
            "Encodings": encodings,
            "Charges": charges,
        }
    )

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
            stripped_line = line.strip()
            if len(stripped_line) == 0:
                if len(spectrum_section) > 0:
                    yield _parse_spectra_sptxt(spectrum_section, *args, **kwargs)
                    spectrum_section = []
            else:
                spectrum_section.append(stripped_line)

        if len(spectrum_section) > 0:
            yield _parse_spectra_sptxt(spectrum_section, *args, **kwargs)


def _parse_spectra_sptxt(x, *args, **kwargs):
    """
    Parses a single spectra into an object

    Meant for internal use
    """
    assert x[0].startswith("Name"), print(x)
    assert x[1].startswith("Comment"), print(x)
    assert x[2].startswith("Num peaks"), print(x)

    name_sec = x[0][x[0].index(":") + 2 :]
    comment_sec = x[1][x[1].index(":") :]
    peaks_sec = x[3:]

    comment_sec = comment_sec.split(" ")
    comment_dict = {e.split("=")[0]: e.split("=")[1] for e in comment_sec if "=" in e}
    sequence, charge = name_sec.split("/")
    sequence = sequence.strip()

    peaks_sec = [l.split() for l in peaks_sec if "." in l]
    mz = [float(l[0]) for l in peaks_sec]
    intensity = [float(l[1]) for l in peaks_sec]

    out_spec = Spectrum(
        sequence=sequence,
        charge=int(charge),
        parent_mz=float(comment_dict["Parent"]),
        intensities=intensity,
        mzs=mz,
        modifications=comment_dict["Mods"],
        *args,
        **kwargs,
    )

    return out_spec
