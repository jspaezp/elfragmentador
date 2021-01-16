
from argparse import ArgumentParser
from transprosit import spectra

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument(
    "--num_spectra",
    type=int,
    default=1e8,
    help="Maximum number of spectra to read",
)

args = parser.parse_args()
dict_args = vars(args)
# Data has to be downloaded independently

specs = spectra.encode_sptxt(
    "/home/jspaezp/Downloads/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-002e0dce-download_filtered_sptxt_library-main.sptxt",
    max_spec=args.num_spectra,
)
specs.to_csv("encoded_spectra.csv", index=False)
