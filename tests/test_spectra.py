from transprosit import spectra


def test_parse_spec():
    sample_spec = [
        "Name: AAAPRPPVSAASGRPQDDTDSSR/3",
        "Comment: Parent=770.3789672852 Mods=0",
        "Num peaks: 23",
        '226.11749267578125      1388.6038503858358      "?"',
        '349.183349609375        6926.61467621213        "?"',
        '467.2710876464844       2084.733731525135       "?"',
    ]

    spec = spectra._parse_spectra_sptxt(sample_spec)
    print(spec)
    spec.annotate_peaks()
    print(spec)
    print(spec.encode_annotations())
    print(spec.encode_annotations(dry=True))

def test_parse_sptxt(shared_datadir):
    print(list(spectra.read_sptxt(str(shared_datadir / "sample.sptxt"))))

def test_spectrum():
    spec = spectra.Spectrum("AAA", 2, 500, mzs=[100.1, 100.2], intensities=[200, 500])
    print(spec)

if __name__ == "__main__":
    test_spectrum()
    test_parse_spec()
