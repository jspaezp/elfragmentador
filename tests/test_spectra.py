import pandas as pd

from elfragmentador import annotate, constants, spectra


def test_parse_spec():
    sample_spec = [
        "Name: AAAPRPPVSAASGRPQDDTDSSR/3",
        "Comment: Parent=770.3789672852 Mods=0",
        "Num peaks: 23",
        '226.11749267578125      1388.6038503858358      "?"',
        '349.183349609375        6926.61467621213        "?"',
        '467.2710876464844       2084.733731525135       "?"',
    ]

    spec = spectra.SptxtReader._parse_spectra_sptxt(sample_spec)
    spec.annotated_peaks
    out_encoding = spec.encode_spectra()
    print(spec)
    print(out_encoding)
    assert len(out_encoding) == constants.NUM_FRAG_EMBEDINGS


def test_parse_spectrast():
    sample_spec = [
        "Name: ASTSDYQVISDR/2",
        "LibID: 0",
        "MW: 1342.6354",
        "PrecursorMZ: 671.3177",
        "Status: Normal",
        "FullName: K.ASTSDYQVISDR.Q/2 (HCD)",
        "Comment: AvePrecursorMz=671.7060 BinaryFileOffset=401 CollisionEnergy=28.0 FracUnassigned=0.67,3/5;0.43,8/20;0.47,220/349 MassDiff=0.0012 Mods=0 NAA=12 NMC=0 NTT=2 Nreps=1/1 OrigMaxIntensity=2.1e+06 Parent=671.318 Pep=Tryptic PrecursorIntensity=5.1e+07 Prob=1.0000 Protein=1/sp|Q8NFH5|NUP35_HUMAN RawSpectrum=20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_60min_15000_02.11507.11507 RetentionTime=600.1,600.1,600.1 Sample=1/_data_interact-20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_60min_15000_02,1,1 Se=1^C1:pb=1.0000/0,fv=4.8531/0 Spec=Raw TotalIonCurrent=2.7e+07",
        "NumPeaks: 349",
        "101.0715\t343.6\tIQA/0.001\t",
        "101.1078\t54.2\t?\t",
        "102.0550\t95.3\tIQAi/-0.016\t",
        "104.0529\t65.8\t?\t",
        "157.1081\t193.5\ty1-18/-0.000\t",
        "158.0927\t377.0\ty1-17/0.000\t",
        "159.0621\t37.3\t?\t",
        "159.0768\t1133.7\tb2/0.000\t",
        "159.0910\t106.1\ty1-17i/-0.001\t",
        "162.0921\t64.1\t?\t",
        "166.0613\t44.3\t?\t",
        "167.0800\t38.1\ty3-44^2/-0.018\t",
        "462.1841\t246.8\tb5/0.001\t",
        "473.2334\t80.0\ty4-17/-0.002\t",
        "490.1970\t109.1\t?\t",
        "490.2626\t3638.8\ty4/0.001\t",
    ]
    spec = spectra.SptxtReader._parse_spectra_sptxt(sample_spec)
    spec.annotated_peaks
    out_encoding = spec.encode_spectra()
    print(spec)
    print(out_encoding)
    assert len(out_encoding) == constants.NUM_FRAG_EMBEDINGS


def test_parse_phospho_spectrast():
    sample_spec = [
        "Name: AAAT[181]PAKKTVT[181]PAK/3",
        "LibID: 0",
        "MW: 1516.7525",
        "PrecursorMZ: 505.5842",
        "Status: Normal",
        "FullName: K.AAAT[181]PAKKTVT[181]PAK.A/3 (HCD)",
        "Comment: AvePrecursorMz=505.8650 BinaryFileOffset=620 CollisionEnergy=28.0 FracUnassigned=0.65,3/5;0.61,12/20;0.54,93/150 MassDiff=0.0003 Mods=2/3,T,Phospho/10,T,Phospho NAA=14 NMC=2 NTT=2 Nreps=1/1 OrigMaxIntensity=9.5e+05 Parent=505.584 Pep=Tryptic PrecursorIntensity=3.2e+07 Prob=0.9967 Protein=1/sp|P19338|NUCL_HUMAN RawSpectrum=20171122_QE3_nLC7_AH_LFQrep2_short90_C2.03163.03163 RetentionTime=605.4,605.4,605.4 Sample=1/_data_interact-20171122_QE3_nLC7_AH_LFQrep2_short90_C2,1,1 Se=1^C1:pb=0.9967/0,fv=2.0933/0 Spec=CONSENSUS TotalIonCurrent=1.8e+07",
        "NumPeaks: 150",
        "101.0713\t299.0\ty2-17^2/0.006\t",
        "143.0816\t5772.5\tb2/0.000,m2:3/0.000\t",
        "147.1129\t2537.1\ty1/0.000\t",
        "158.0924\t365.8\ty3^2/-0.013,b4-80^2/0.006\t",
        "580.3473\t350.3\ty6-116/0.002\t",
        "595.2847\t303.2\ty5/-0.000\t",
        "824.4262\t405.8\ty7/-0.002\t",
        "952.5225\t271.0\ty8/-0.000\t",
    ]
    spec = spectra.SptxtReader._parse_spectra_sptxt(sample_spec)
    spec.annotated_peaks
    print(spec)
    out_encoding = spec.encode_spectra()
    print(spec.annotated_peaks)
    print(spec._theoretical_peaks)
    assert len(out_encoding) == constants.NUM_FRAG_EMBEDINGS
    assert sum(x > 0 for x in out_encoding) == 5
    out_encoding = spec.encode_sequence()
    print(out_encoding)
    assert len(out_encoding.aas) == constants.MAX_TENSOR_SEQUENCE
    assert len(out_encoding.mods) == constants.MAX_TENSOR_SEQUENCE


def test_parse_sptxt(shared_datadir):
    in_path = str(shared_datadir / "sample.sptxt")
    print(list(spectra.SptxtReader(in_path)))


def test_parse_spectrast_sptxt(shared_datadir):
    in_path = str(shared_datadir / "small_proteome_spectrast.sptxt")
    print(list(spectra.SptxtReader(in_path)))


def test_parse_phospho_spectrast_sptxt(shared_datadir):
    in_path = str(shared_datadir / "small_phospho_spectrast.sptxt")
    print(list(spectra.SptxtReader(in_path)))


def test_benchmark_spectra_parsing(shared_datadir, benchmark):
    in_path = str(shared_datadir / "single_spectrum.txt")
    with open(in_path, "r") as f:
        spec_chunk = list(f)
    spec = spectra.SptxtReader._parse_spectra_sptxt(spec_chunk)
    out = benchmark(
        annotate.annotate_peaks, spec._theoretical_peaks, spec.mzs, spec.intensities
    )
    print(out)


def test_spectrum():
    spec = spectra.Spectrum(
        "AAA", 2, 500, mzs=[100.1, 100.2], intensities=[200, 500], nce=27.0
    )
    print(spec)


def test_spectrum_works_on_term_acetyl():
    spec = spectra.Spectrum(
        "n[43]AAA", 2, 500, mzs=[100.1, 100.2], intensities=[200, 500], nce=27.0
    )
    print(spec)


def test_sptxt_to_pd(shared_datadir):
    in_path = str(shared_datadir / "small_phospho_spectrast.sptxt")
    df = spectra.SptxtReader(in_path).to_df()

    assert isinstance(df, pd.DataFrame)


if __name__ == "__main__":
    test_spectrum()
    test_parse_spec()
