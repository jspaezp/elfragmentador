from transprosit import spectra

# Data has to be downloaded independently

specs = spectra.encode_sptxt(
    "/home/jspaezp/Downloads/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-002e0dce-download_filtered_sptxt_library-main.sptxt"
)
specs.to_csv("encoded_spectra.csv", index=False)

specs = spectra.read_sptxt(
    "~/Downloads/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-002e0dce-download_filtered_sptxt_library-main.sptxt"
)
spec = next(specs)
spec
