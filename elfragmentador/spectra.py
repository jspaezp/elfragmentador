"""
Contains utilities to represent spectra as well as functions to read them in bulk from
.sptxt files
"""

from __future__ import annotations
import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import warnings
from typing import Iterator, Dict, Optional, List, Sequence, Union
from pathlib import Path

from elfragmentador import constants as CONSTANTS
from elfragmentador import annotate, encoding_decoding, scoring
import elfragmentador
from elfragmentador.encoding_decoding import get_fragment_encoding_labels, SequencePair

from pandas.core.frame import DataFrame
import numpy as np
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

from tqdm.auto import tqdm


class Spectrum:
    """Represents Spectra and bundles methods to annotate peaks."""

    __SPTXT_TEMPLATE = (
        "Name: {name}\n"
        # "LibID: {lib_id}\n"
        "MW: {mw}\n"
        "PrecursorMZ: {precursor_mz}\n"
        "FullName: {full_name}\n"
        "Comment: {comment}\n"
        "Num Peaks: {num_peaks}\n"
        "{peaks}\n\n"
    )

    def __init__(
        self,
        sequence: str,
        charge: int,
        parent_mz: Union[float, int],
        mzs: Union[List[float], List[int]],
        intensities: Union[List[float], List[int]],
        nce: Optional[float],
        modifications: Optional[str] = None,
        instrument: None = None,
        analyzer: str = "FTMS",
        rt: Optional[float] = None,
        irt: Optional[float] = None,
        raw_spectra: Optional[str] = None,
        nreps: Optional[int] = None,
    ) -> None:
        """
        Representation of spectra with methods to convert from and to encodings.

        This class provides a way to represent spectra and its associated peptide sequence,
        as well as multiple methods to convert these information to the encoding required
        for training the main model

        Parameters
        ----------
        sequence : str
            String representing the aminoacid sequence with or without modifications
        charge : int
            Charge of the precursor associated with the spectrum
        parent_mz : float
            Observed m/z of the precursor
        mzs : List[float]
            Iterable of the observed masses of the fragment ions
        intensities : List[float]
            Iterable of the observed intensities of the fragment ions, should
            match in length with the length of `mzs`
        nce : float
            Collision Energy used during the fragmentation.
        modifications : str, optional
            Currently unused string describing the modifications, by default None
        instrument : str, optional
            Currently unused instrument where the spectra was collected, by default None
        analyzer : str, optional
            Either of 'FTMS', 'ITMS', 'TripleTOF', despite the annotation working on all
            of them, the model currently supports only FTMS data, by default FTMS
        rt : float, optional
            Retention time of the spectra, by default None
        raw_spectra : str, optional
            String describing the file where the spectra originated, by default None
        nreps : int, optional
            Integer describing how many spectra were used to generate this concensus spectrum
        """
        tolerance, tolerance_unit = CONSTANTS.TOLERANCE[analyzer]
        parsed_peptide = list(annotate.peptide_parser(sequence, solve_aliases=True))

        # Makes sure all elements in the sequence are aminoacids
        assert set(parsed_peptide) <= CONSTANTS.AMINO_ACID_SET.union(
            CONSTANTS.MOD_PEPTIDE_ALIASES
        ), f"Assertion of supported modifications failed for {sequence}: {parsed_peptide}"
        sequence = "".join(parsed_peptide)
        self.sequence = "".join([x[:1] for x in parsed_peptide])
        self.mod_sequence = sequence
        self.length = len(encoding_decoding.clip_explicit_terminus(parsed_peptide))
        self.charge = charge
        self.parent_mz = parent_mz
        self.modifications = modifications

        amino_acids = list(annotate.peptide_parser(sequence))

        # TODO consider if this masses should be calculated in a lazy manner
        # TODO redefine these with the functions inside annotate
        self.theoretical_mass = annotate.get_theoretical_mass(amino_acids)
        self.theoretical_mz = (
            self.theoretical_mass + (charge * CONSTANTS.PROTON)
        ) / charge

        assert len(mzs) == len(intensities)
        self.mzs = mzs
        self.intensities = intensities

        self.delta_m = abs(self.parent_mz - self.theoretical_mz)
        self.delta_ppm = 1e6 * abs(self.delta_m) / self.theoretical_mz

        # Annotation related section
        self.tolerance = tolerance
        self.tolerance_unit = tolerance_unit

        # Dict with ION_NAME: ION MASS
        self._theoretical_peaks = annotate.get_peptide_ions(self.mod_sequence)

        self._annotated_peaks = None
        self._delta_ascore = None
        self.nce = nce
        self.instrument = instrument
        self.analyzer = analyzer
        self.rt = rt
        self.irt = irt
        self.raw_spectra = raw_spectra
        self.nreps = nreps

    @classmethod
    def theoretical_spectrum(
        cls,
        seq: str,
        charge: int,
    ) -> Spectrum:
        """theoretical_spectrum Generates theoretical spectra from sequences.

        Parameters
        ----------
        seq : str
            Peptide sequence
        charge : int
            Precursor charge to use

        Returns
        -------
        Spectrum
            A spectrum object with 1 as the theoretical intensities

        Examples
        --------
        >>> spec = Spectrum.theoretical_spectrum("MYPEPTIDE", 2)
        >>> spec
        Spectrum:
            Sequence: MYPEPTIDE len:9
            Mod.Sequence: MYPEPTIDE
            Charge: 2
            MZs: [132.047761467, 148.060434167, 263.087377167]...16
            Ints: [1.0, 1.0, 1.0]...16
            Instrument: None
            Analyzer: FTMS
            NCE: None
            RT: None
            OriginalSpectra: Predicted
            Annotations: {'z2y2': 1.0, 'z2b2': 1.0, ...
        """
        ions = annotate.get_peptide_ions(seq)
        ions = {k: v for k, v in ions.items() if int(k[1]) < charge}
        parent_mz = annotate.get_precursor_mz(seq, charge)
        mzs = list(ions.values())
        mzs = sorted(mzs)
        intensities = [1.0 for _ in mzs]

        spec = cls(
            sequence=seq,
            charge=charge,
            parent_mz=parent_mz,
            mzs=mzs,
            intensities=intensities,
            nce=None,
            raw_spectra="Predicted",
        )

        spec.annotated_peaks

        return spec

    @classmethod
    def from_tensors(
        cls,
        sequence_tensor: List[int],
        fragment_tensor: List[int],
        mod_tensor: None = None,
        charge: int = 2,
        nce: float = 27.0,
        parent_mz: int = 0,
        *args,
        **kwargs,
    ) -> Spectrum:
        """
        from_tensors Encodes iterables into a Spectrum object.

        This method is an utility function to create Spectrum objects from the encoded
        iterables or tensors. The encoding entail two iterables of integers,
        the sequence and the fragment (optionally the modifications).

        For the values of the encoding please visit the constants submodule

        Parameters
        ----------
        sequence_tensor : List[int]
            [description]
        fragment_tensor : List[float]
            [description]
        mod_tensor : List[int], optional
            [description], by default None
        charge: int
        nce: float
            Normalized collision energy

        Returns
        -------
        Spectrum
            A spectrum object decoding the sequences

        Examples
        --------
        >>> Spectrum.from_tensors([1, 1, 2, 3, 0, 0, 0, 0, 0, 0], [0]*CONSTANTS.NUM_FRAG_EMBEDINGS)
        Spectrum:
            Sequence: AACD len:4
            Mod.Sequence: AACD
            Charge: 2
            MZs: [72.044390467, 134.044784167, 143.081504467]...18
            Ints: [0.0, 0.0, 0.0]...18
            Instrument: None
            Analyzer: FTMS
            NCE: 27.0
            RT: None
            OriginalSpectra: None
        """
        mod_sequence = encoding_decoding.decode_mod_seq(sequence_tensor, mod_tensor)
        fragment_df = encoding_decoding.decode_fragment_tensor(
            mod_sequence, fragment_tensor
        )
        self = cls(
            mod_sequence,
            mzs=[float(x) for x in fragment_df["Mass"]],
            intensities=[float(x) for x in fragment_df["Intensity"]],
            charge=charge,
            parent_mz=parent_mz,
            nce=nce,
            *args,
            **kwargs,
        )
        return self

    def precursor_error(self, error_type: str = "ppm") -> float:
        """
        precursor_error Calculates the mass error of the precursor.

        Calculates the mass error of the precursor, knowing the sequence,
        and modifications, in addition to the observed mass

        Parameters
        ----------
        error_type : ppm or da, optional
            The type of mass error that will be calculated, by default "ppm"

        Returns
        -------
        float
            the mass error...

        Raises
        ------
        NotImplementedError
            Raises this error if any error type other than ppm or da is provided

        Examples
        --------
        >>> myspec = Spectrum("AA", charge=1, parent_mz=161.0920, mzs=[101.0713], intensities=[299.0], nce = 27.0, )
        >>> myspec.precursor_error("ppm")
        0.42936316076909053
        >>> myspec = Spectrum("AAAT[181]PAKKTVT[181]PAK", charge=3, parent_mz=505.5842, mzs=[101.0713, 143.0816, 147.1129], intensities=[299.0, 5772.5, 2537.1], nce = 27.0, )
        >>> myspec.precursor_error("ppm")
        0.06981956539363246
        >>> myspec.precursor_error("da")
        3.529966664927997e-05
        """
        if error_type == "ppm":
            return self.delta_ppm
        elif error_type == "da":
            return self.delta_m
        else:
            raise NotImplementedError(
                "Not a know error type, select either of 'ppm' or 'da'"
            )

    def _annotate_peaks(self) -> None:
        annots = annotate.annotate_peaks(
            self._theoretical_peaks,
            self.mzs,
            self.intensities,
            self.tolerance,
            self.tolerance_unit,
        )
        assert (
            len(annots) > 0
        ), f"No peaks were annotated in this spectrum {self.sequence}"
        if len(annots) < 3:
            warnings.warn(
                f"Less than 3 ({len(annots)}) peaks were"
                f" annotated for spectra {self.sequence}"
            )

        self._annotated_peaks = annots

    def _calculate_delta_ascore(self) -> None:
        self._delta_ascore = scoring.calc_delta_ascore(
            seq=encoding_decoding.clip_explicit_terminus(self.mod_sequence),
            mod=list(CONSTANTS.VARIABLE_MODS.keys()),
            aas=list(CONSTANTS.VARIABLE_MODS.values()),
            mzs=self.mzs,
            ints=self.intensities,
        )

    @property
    def delta_ascore(self) -> float:
        if self._delta_ascore is None:
            self._calculate_delta_ascore()

        return self._delta_ascore

    def encode_spectra(
        self, dry: bool = False
    ) -> Union[List[Union[int, float]], List[str]]:
        """
        encode_spectra Produce encoded sequences from your spectrum object.

        It produces a list of integers that represents the spectrum, the labels correspond
        to the ones in CONSTANTS.FRAG_EMBEDING_LABELS, but can also be acquired using the
        argument dry=True

        Parameters
        ----------
        dry : bool, optional
            wether to actually compute the encoding or only return the labels, by default False

        Returns
        -------
        List[int] or List[str]
            The list of intensities matching the ions (normalized to the highest) or
            the labels for such ions, depending on wether the dry argument was passed or not


        Examples
        --------
        >>> myspec = Spectrum("AAAT[181]PAKKTVT[181]PAK", charge=3, parent_mz=505.5842, mzs=[101.0713, 143.0816, 147.1129], intensities=[299.0, 5772.5, 2537.1], nce = 27.0)
        >>> myspec.encode_spectra()
        [0, 0.4395149415331312, 1.0, 0, 0, 0, 0, ..., 0]
        >>> len(myspec.encode_spectra())
        174
        >>> myspec.encode_spectra(dry=True)
        ['z1b1', 'z1y1', 'z1b2', 'z1y2', 'z1b3',..., 'z3b29', 'z3y29']
        """
        if self._annotated_peaks is None and not dry:
            self.annotated_peaks
            self.num_matching_peaks = sum(
                [1 for x in self.annotated_peaks.values() if x > 0]
            )

        if dry:
            peak_annot = None
        else:
            peak_annot = self.annotated_peaks

        return get_fragment_encoding_labels(annotated_peaks=peak_annot)

    def encode_sequence(self) -> SequencePair:
        """
        encode_sequence returns the encoded sequence of the aminoacids/modifications.

        It returns two lists representing the aminoacid and modification sequences, the
        length of the sequence will correspond to CONSTANTS.MAX_TENSOR_SEQUENCE.

        The meaning of each corresponding index comes from CONSTANTS.ALPHABET and
        CONSTANTS.MOD_INDICES. Some aliases for modifications are supported, check them
        at constans.MOD_PEPTIDE_ALIASES

        Returns
        -------
        Tuple[List[int], List[int]]
            A named tuple with `aas` and `mods` as names, containging respectively the
            encoding of aminoacids and modifications.

        Examples
        --------
        >>> myspec = Spectrum("AAAT[181]PAKKTVT[181]PAK", charge=3, parent_mz=505.5842, mzs=[101.0713, 143.0816, 147.1129], intensities=[299.0, 5772.5, 2537.1], nce = 27.0)
        >>> myspec.encode_sequence()
        SequencePair(aas=[23, 1, 1, 1, 17, 13, 1, 9, 9, 17, 19, ..., 0], mods=[0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, ..., 0])
        """
        return encoding_decoding.encode_mod_seq(self.mod_sequence)

    @property
    def annotated_peaks(self) -> Dict[str, float]:
        """
        annotated_peaks Peaks in the spectra annotated as ions.

        Contains the annotated ions and its corresponding intensity, normalized
        to the highest one in the spectra

        Returns
        -------
        dict
            Keys are the charge and ion types (z1y1) and the values
            are the normalized intensity of the ion.

        Examples
        --------
        >>> myspec = Spectrum("AAAT[181]PAKKTVT[181]PAK", charge=3, parent_mz=505.5842, mzs=[101.0713, 143.0816, 147.1129], intensities=[299.0, 5772.5, 2537.1], nce = 27.0)
        >>> myspec.annotated_peaks
        {'z1b2': 1.0, 'z1y1': 0.4395149415331312}
        """
        if self._annotated_peaks is None:
            self._annotate_peaks()

        return self._annotated_peaks

    def __repr__(self) -> str:
        """
        __repr__ Represents the summary of the spectrum for the console.

        it is implicitly called by print but allows nice printing of the Spectrum
        objects in the console for debugging purposes mainly.

        Returns
        -------
        str
            String representation of the object

        Examples
        --------
        >>> myspec = Spectrum("MYPEPT[181]IDEK", 2, 200, [100, 200], [1e8, 1e7], nce=27.0)
        >>> myspec
        Spectrum:
            Sequence: MYPEPTIDEK len:10
            Mod.Sequence: MYPEPT[PHOSPHO]IDEK
            Charge: 2
            MZs: [100, 200]...2
            Ints: [100000000.0, 10000000.0]...2
            Instrument: None
            Analyzer: FTMS
            NCE: 27.0
            RT: None
            OriginalSpectra: None
        """
        out = (
            "Spectrum:\n"
            f"\tSequence: {encoding_decoding.clip_explicit_terminus(self.sequence)}"
            f" len:{self.length}\n"
            f"\tMod.Sequence: {encoding_decoding.clip_explicit_terminus(self.mod_sequence)}\n"
            f"\tCharge: {self.charge}\n"
            f"\tMZs: {self.mzs[:3]}...{len(self.mzs)}\n"
            f"\tInts: {self.intensities[:3]}...{len(self.intensities)}\n"
            f"\tInstrument: {self.instrument}\n"
            f"\tAnalyzer: {self.analyzer}\n"
            f"\tNCE: {self.nce}\n"
            f"\tRT: {self.rt}\n"
            f"\tOriginalSpectra: {self.raw_spectra}\n"
        )

        if self._annotated_peaks is not None:
            out += f"\tAnnotations: {self._annotated_peaks}\n"

        return out

    def to_sptxt(self) -> str:
        """
        to_sptxt Represents the spectrum for an sptxt file

        Returns
        -------
        str
            String representation of the object

        Examples
        --------
        >>> myspec = Spectrum("MYPEPT[181]IDEK", 2, 200, [100, 200], [1e8, 1e7], nce=27.0)
        >>> myspec
        Spectrum:
            Sequence: MYPEPTIDEK len:10
            Mod.Sequence: MYPEPT[PHOSPHO]IDEK
            Charge: 2
            MZs: [100, 200]...2
            Ints: [100000000.0, 10000000.0]...2
            Instrument: None
            Analyzer: FTMS
            NCE: 27.0
            RT: None
            OriginalSpectra: None
        >>> print(myspec.to_sptxt())
        Name: MYPEPT[80]IDEK/2
        MW: 1301.5250727
        PrecursorMZ: 651.769812817
        FullName: MYPEPT[80]IDEK/2 (HCD)
        Comment: CollisionEnergy=27.0 ...
        Num Peaks: 2
        100\t100000000.0\t"?"
        200\t10000000.0\t"?"


        """
        mod_seq = annotate.mass_diff_encode_seq(self.mod_sequence)
        name = f"{mod_seq}/{self.charge}"
        mw = self.theoretical_mass
        precursor_mz = self.theoretical_mz
        full_name = name + " (HCD)"
        comment = {
            "CollisionEnergy": self.nce,
            "Origin": f"ElFragmentador_v{elfragmentador.__version__}",
        }

        if self.rt is not None:
            comment.update({"RetentionTime": self.rt})

        if self.irt is not None:
            comment.update({"iRT": self.irt})

        comment = " ".join([f"{k}={v}" for k, v in comment.items()])
        peak_list = [
            f'{x}\t{y}\t"?"' for x, y in zip(self.mzs, self.intensities) if y > 0.001
        ]

        peaks = "\n".join(peak_list)

        out = self.__SPTXT_TEMPLATE.format(
            name=name,
            mw=mw,
            precursor_mz=precursor_mz,
            full_name=full_name,
            comment=comment,
            num_peaks=len(peak_list),
            peaks=peaks,
        )
        return out

    @property
    def sus_msms_spec(self) -> sus.MsmsSpectrum:
        if not hasattr(self, "_sus_msms_spec"):
            self._sus_msms_spec = self._to_spectrum_utils()

        return self._sus_msms_spec

    def _to_spectrum_utils(self):
        aas, mods = self.encode_sequence()
        TERM_ALIAS_DICT = {
            CONSTANTS.ALPHABET["c"]: "C-term",
            CONSTANTS.ALPHABET["n"]: "N-term",
        }

        modifications = {
            (
                (i - 1) if aa not in TERM_ALIAS_DICT else TERM_ALIAS_DICT[aa]
            ): CONSTANTS.MODIFICATION[CONSTANTS.MOD_INDICES_S[m]]
            for i, (m, aa) in enumerate(zip(mods, aas))
            if m != 0
        }

        modifications.update(
            {
                i - 1: CONSTANTS.MODIFICATION["CARBAMIDOMETHYL"]
                for i, (m, aa) in enumerate(zip(mods, aas))
                if aa == CONSTANTS.ALPHABET["C"] and m == 0
            }
        )

        stripped_sequence = encoding_decoding.clip_explicit_terminus(self.sequence)
        # TODO consider if this is needed ...
        # and (sus._aa_mass["C"] - 103.00919) < 0.0001
        spectrum = sus.MsmsSpectrum(
            identifier=stripped_sequence,
            precursor_mz=self.parent_mz,
            precursor_charge=self.charge,
            mz=self.mzs,
            intensity=self.intensities,
            retention_time=self.rt,
            peptide=stripped_sequence,
            modifications=modifications,
        )

        # Process the MS/MS spectrum.
        tolerance, tolerance_unit = CONSTANTS.TOLERANCE[self.analyzer]
        spectrum = spectrum.filter_intensity(
            min_intensity=0.005
        ).annotate_peptide_fragments(
            tolerance,
            tolerance_unit,
            ion_types=CONSTANTS.ION_TYPES,
            max_ion_charge=self.charge,
        )

        return spectrum

    def plot(self, mirror: Union[Spectrum, sus.MsmsSpectrum] = None, ax=None, **kwargs):
        if mirror is None:
            sup.spectrum(self.sus_msms_spec, ax=ax, **kwargs)
        else:
            if isinstance(mirror, Spectrum):
                mirror = mirror.sus_msms_spec
            sup.mirror(
                spec_top=self.sus_msms_spec,
                spec_bottom=mirror,
                spectrum_kws=kwargs,
                ax=ax,
            )


def encode_sptxt(
    filepath: Union[str, Path],
    max_spec: float = 1e9,
    min_peaks: int = 3,
    min_delta_ascore: int = 20,
    irt_fun: None = None,
    *args,
    **kwargs,
) -> DataFrame:
    """
    encode_sptxt Convert an sptxt file to a dataframe containing encodings.

    Converts the spectra contained in a .sptxt file to a pandas DataFrame that
    contains the relevant fields required to train the main model.

    Parameters
    ----------
    filepath : Path or str
        Path of the .sptxt file to read
    max_spec : int, optional
        Maximum number of spectra to read, by default 1e9
    min_peaks : int
        Minimum number of annotated peaks for a spectrum to be added
    irt_fun : [type], optional
        Not yet implemented but would take a callable that converts
        the retention times to iRTs, by default None

    Returns
    -------
    DataFrame
        DataFrame containing the data required to train the model
        and would be taken by the PeptideDataset
        # TODO specify the columns

    Raises
    ------
    NotImplementedError
        Raises this error when an iRT converter function is passed
        because I have not implemented it....
    """
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
    d_ascores = []
    nreps = []

    i = 0
    skipped_spec = 0
    for spec in tqdm(iter):
        i += 1
        if i >= max_spec:
            break

        # TODO add offset to skip the first x sequences and a way to make the selection random
        seq_encode, mod_encode = spec.encode_sequence()
        seq_encode, mod_encode = str(seq_encode), str(mod_encode)

        try:
            spec_encode = spec.encode_spectra()
            spec_encode = [round(x, 5) for x in spec_encode]
            spec_encode = str(spec_encode)
        except AssertionError as e:
            warnings.warn(f"Skipping because of error: {e}")
            skipped_spec += 1
            continue

        if min_peaks is not None and spec.num_matching_peaks < min_peaks:
            warnings.warn(
                f"Skipping peptide due few peaks being annotated {spec.mod_sequence}"
            )
            skipped_spec += 1
            continue

        if min_delta_ascore is not None and spec.delta_ascore < min_delta_ascore:
            warnings.warn(
                f"Skipping peptide due low ascore '{spec.delta_ascore}' {spec.mod_sequence}"
            )
            skipped_spec += 1
            continue

        spectra_encodings.append(spec_encode)
        seq_encodings.append(seq_encode)
        mod_encodings.append(mod_encode)
        charges.append(spec.charge)
        sequences.append(spec.sequence)
        mod_sequences.append(spec.mod_sequence)
        rts.append(spec.rt)
        nces.append(spec.nce)
        orig.append(spec.raw_spectra)
        d_ascores.append(spec.delta_ascore)
        nreps.append(spec.nreps)

    ret = DataFrame(
        {
            "Sequences": sequences,
            "ModSequences": mod_sequences,
            "Charges": charges,
            "NCEs": nces,
            "RTs": rts,
            "SpectraEncodings": spectra_encodings,
            "ModEncodings": mod_encodings,
            "SeqEncodings": seq_encodings,
            "OrigSpectra": orig,
            "DeltaAscore": d_ascores,
            "Nreps": nreps,
        }
    )

    if irt_fun is not None:
        raise NotImplementedError
    else:
        """
        warnings.warn(
            "No calculation function passed for iRT,"
            " will replace the column with missing"
        )
        """
        ret["iRT"] = np.nan

    if skipped_spec >= 1:
        warnings.warn(f"{skipped_spec}/{i} Spectra were skipped")

    logging.info(list(ret))
    logging.info(ret)

    return ret


def sptxt_to_csv(filepath, output_path, filter_irt_peptides=True, *args, **kwargs):
    df = encode_sptxt(filepath=filepath, *args, **kwargs)
    if filter_irt_peptides:
        df = df[[x not in CONSTANTS.IRT_PEPTIDES for x in df["Sequences"]]]
    df.to_csv(output_path, index=False)


# TODO consider if moving this parser to just use another dependency ... pyteomics ??
def read_sptxt(filepath: str, *args, **kwargs) -> Iterator[Spectrum]:
    """
    read_sptxt reads a spectra library file.

    reads a spectral library file into a list of spectra objects

    Parameters
    ----------
    filepath : Path or str
        The path to the spectral library, extension .sptxt
    *args
        Passed onto Spectrum
    **kwargs
        Passed onto Spectrum

    Yields
    ------
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
                    try:
                        yield _parse_spectra_sptxt(spectrum_section, *args, **kwargs)
                    except AssertionError as e:
                        warnings.warn(f"Skipping spectra with assertion error: {e}")
                        pass
                    spectrum_section = []
            else:
                spectrum_section.append(stripped_line)

        if len(spectrum_section) > 0:
            try:
                yield _parse_spectra_sptxt(spectrum_section, *args, **kwargs)
            except AssertionError as e:
                warnings.warn(f"Skipping spectra with assertion error: {e}")


def _parse_spectra_sptxt(
    x: List[str], instrument: None = None, analyzer: str = "FTMS", *args, **kwargs
) -> Spectrum:
    """
    Parse a single spectra into an object.

    Meant for internal use
    """
    digits = [str(v) for v in range(10)]

    # Header Handling
    named_params = [v for v in x if v[:1].isalpha() and ":" in v]
    named_params_dict = {}
    for v in named_params:
        tmp = v.split(":")
        named_params_dict[tmp[0].strip()] = ":".join(tmp[1:])

    fragmentation = named_params_dict.get("FullName", None)
    if fragmentation is not None:
        fragmentation = fragmentation[fragmentation.index("(") + 1 : -1]

    comment_sec = [
        v.split("=") for v in named_params_dict["Comment"].strip().split(" ")
    ]
    comment_dict = {v[0]: v[1] for v in comment_sec}
    sequence, charge = named_params_dict["Name"].split("/")

    nce = comment_dict.get("CollisionEnergy", None)
    if nce is not None:
        nce = float(nce)

    rt = comment_dict.get("RetentionTime", None)
    if rt is not None:
        rt = float(rt.split(",")[0])

    irt = comment_dict.get("iRT", None)
    if irt is not None:
        irt = float(irt.split(",")[0])

    nreps = comment_dict.get("Nreps", None)
    if nreps is not None:
        nreps = int(nreps.split("/")[0])

    raw_spectra = comment_dict.get("RawSpectrum", None) or comment_dict.get(
        "BestRawSpectrum", None
    )

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
        analyzer=analyzer,
        instrument=instrument,
        nce=nce,
        rt=rt,
        irt=irt,
        raw_spectra=raw_spectra,
        nreps=nreps,
        *args,
        **kwargs,
    )

    return out_spec
