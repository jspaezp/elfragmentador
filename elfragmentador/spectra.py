"""
Contains utilities to represent spectra as well as functions to read them in bulk from
.sptxt files
"""

import warnings
from typing import Optional, List, Union
from pathlib import Path

from elfragmentador import constants
from elfragmentador import annotate
from elfragmentador import encoding_decoding
from elfragmentador.encoding_decoding import get_fragment_encoding_labels

from pandas.core.frame import DataFrame
import numpy as np
from tqdm.auto import tqdm


class Spectrum:
    """Represents Spectra and bundles methods to annotate peaks."""

    def __init__(
        self,
        sequence: str,
        charge: int,
        parent_mz: float,
        mzs: List[float],
        intensities: List[float],
        nce: float,
        modifications=None,
        instrument=None,
        analyzer="FTMS",
        rt=None,
        raw_spectra=None,
    ):
        """
        Representation of spectra with methods to conver from and to encodings.

        This class provides a way to represent spectra and its associated peptide sequence,
        as well as multiple methods to convert these information to the encoding required
        for training the main model

        Parameters
        ----------
        sequence : str
            String representing the aminoacid sequence with or without modidications
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
        """
        tolerance, tolerance_unit = constants.TOLERANCE[analyzer]
        parsed_peptide = list(annotate.peptide_parser(sequence))
        if parsed_peptide[0] == "n[43]":
            parsed_peptide.pop(0)
            parsed_peptide[0] += "[nACETYL]"

        # Makes sure all elements in the sequence are aminoacids
        assert set(parsed_peptide) <= constants.AMINO_ACID_SET.union(
            constants.MOD_PEPTIDE_ALIASES
        ), f"Assertion of supported modifications failed for {sequence}: {parsed_peptide}"
        sequence = "".join(parsed_peptide)
        self.sequence = "".join([x[:1] for x in parsed_peptide])
        self.mod_sequence = sequence
        self.length = len(parsed_peptide)
        self.charge = charge
        self.parent_mz = parent_mz
        self.modifications = modifications

        amino_acids = annotate.peptide_parser(sequence)
        self.theoretical_mass = (
            sum([constants.MOD_AA_MASSES[a] for a in amino_acids]) + constants.H2O
        )
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

        self._theoretical_peaks = annotate.get_peptide_ions(self.mod_sequence)

        self._annotated_peaks = None
        self.nce = nce
        self.instrument = instrument
        self.analyzer = analyzer
        self.rt = rt
        self.raw_spectra = raw_spectra

    @staticmethod
    def from_tensors(
        sequence_tensor: List[int],
        fragment_tensor: List[float],
        mod_tensor: List[int] = None,
        charge: int = 2,
        nce: float = 27.0,
        parent_mz: float = 0,
        *args,
        **kwargs,
    ):
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
        >>> Spectrum.from_tensors([1, 1, 2, 3, 0, 0, 0, 0, 0, 0], [0]*constants.NUM_FRAG_EMBEDINGS)
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
        spec_out = Spectrum(
            mod_sequence,
            mzs=[float(x) for x in fragment_df["Mass"]],
            intensities=[float(x) for x in fragment_df["Intensity"]],
            charge=charge,
            parent_mz=parent_mz,
            nce=nce,
            *args,
            **kwargs,
        )
        return spec_out

    def precursor_error(self, error_type="ppm"):
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
        annots = annotate.annotate_peaks2(
            self._theoretical_peaks,
            self.mzs,
            self.intensities,
            self.tolerance,
            self.tolerance_unit,
        )
        self._annotated_peaks = annots

    def encode_spectra(self, dry=False) -> Union[List[int], List[str]]:
        """
        encode_spectra Produce encoded sequences from your spectrum object.

        It produces a list of integers that represents the spectrum, the labels correspond
        to the ones in constants.FRAG_EMBEDING_LABELS, but can also be acquired using the
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

        if dry:
            peak_annot = None
        else:
            peak_annot = self.annotated_peaks

        return get_fragment_encoding_labels(annotated_peaks=peak_annot)

    def encode_sequence(self):
        """
        encode_sequence returns the encoded sequence of the aminoacids/modifications.

        It returns two lists representing the aminoacid and modification sequences, the
        length of the sequence will correspond to constants.MAX_SEQUENCE.

        The meaning of each corresponding index comes from constants.ALPHABET and
        constants.MOD_INDICES. Some aliases for modifications are supported, check them
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
        SequencePair(aas=[1, 1, 1, 17, 13, 1, 9, 9, 17, 19, ..., 0], mods=[0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, ..., 0])
        """
        return encoding_decoding.encode_mod_seq(self.mod_sequence)

    @property
    def annotated_peaks(self):
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

        it is implicitly called by print but allows nice prining of the SPectrum
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
            Mod.Sequence: MYPEPT[181]IDEK
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
            f"\tSequence: {self.sequence} len:{self.length}\n"
            f"\tMod.Sequence: {self.mod_sequence}\n"
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


def encode_sptxt(
    filepath: str, max_spec: float = 1e9, irt_fun: None = None, *args, **kwargs
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

    for i, spec in enumerate(tqdm(iter)):
        if i >= max_spec:
            break
        # TODO add offset to skip the first x sequences
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


def sptxt_to_csv(filepath, output_path, *args, **kwargs):
    df = encode_sptxt(filepath=filepath, *args, **kwargs)
    df.to_csv(output_path, index=False)


def read_sptxt(filepath: Path, *args, **kwargs) -> List[Spectrum]:
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


def _parse_spectra_sptxt(x, instrument=None, analyzer="FTMS", *args, **kwargs):
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

    if comment_dict.get("iRT", None) is not None:
        warnings.warn(
            (
                "Noticed the comment dict has iRT values,"
                " We have not implemented reading them but will do in the future"
            ),
            FutureWarning,
        )

    raw_spectra = comment_dict.get("RawSpectrum", None) or comment_dict.get("BestRawSpectrum", None)

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
        raw_spectra=raw_spectra,
        *args,
        **kwargs,
    )

    return out_spec
