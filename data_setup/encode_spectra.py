from transprosit import spectra

# Data has to be downloaded independently

specs = spectra.encode_sptxt(
    "~/Downloads/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-002e0dce-download_filtered_sptxt_library-main.sptxt"
)
specs.to_csv("encoded_spectra.csv", index=False)

specs = spectra.read_sptxt(
    "/home/jspaezp/Downloads/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-002e0dce-download_filtered_sptxt_library-main.sptxt"
)
spec = next(specs)
spec.encode_annotations(dry=True)
# ['z1b1', 'z1y1', 'z2b1', 'z2y1', 'z3b1', 'z3y1', 'z1b2',
#  'z1y2', 'z2b2', 'z2y2', 'z3b2', 'z3y2', 'z1b3', 'z1y3',
#  'z2b3', 'z2y3', 'z3b3', 'z3y3', 'z1b4', 'z1y4', 'z2b4',
#  'z2y4', 'z3b4', 'z3y4', 'z1b5', 'z1y5', 'z2b5', 'z2y5',
#  'z3b5', 'z3y5', 'z1b6', 'z1y6', 'z2b6', 'z2y6', 'z3b6',
#  'z3y6', 'z1b7', 'z1y7', 'z2b7', 'z2y7', 'z3b7', 'z3y7',
#  'z1b8', 'z1y8', 'z2b8', 'z2y8', 'z3b8', 'z3y8', 'z1b9',
#  'z1y9', 'z2b9', 'z2y9', 'z3b9', 'z3y9', 'z1b10', 'z1y10',
#  'z2b10', 'z2y10', 'z3b10', 'z3y10', 'z1b11', 'z1y11',
#  'z2b11', 'z2y11', 'z3b11', 'z3y11', 'z1b12', 'z1y12',
#  'z2b12', 'z2y12', 'z3b12', 'z3y12', 'z1b13', 'z1y13',
#  'z2b13', 'z2y13', 'z3b13', 'z3y13', 'z1b14', 'z1y14',
#  'z2b14', 'z2y14', 'z3b14', 'z3y14', 'z1b15', 'z1y15',
#  'z2b15', 'z2y15', 'z3b15', 'z3y15', 'z1b16', 'z1y16',
#  'z2b16', 'z2y16', 'z3b16', 'z3y16', 'z1b17', 'z1y17',
#  'z2b17', 'z2y17', 'z3b17', 'z3y17', 'z1b18', 'z1y18',
#  'z2b18', 'z2y18', 'z3b18', 'z3y18', 'z1b19', 'z1y19',
#  'z2b19', 'z2y19', 'z3b19', 'z3y19', 'z1b20', 'z1y20',
#  'z2b20', 'z2y20', 'z3b20', 'z3y20', 'z1b21', 'z1y21',
#  'z2b21', 'z2y21', 'z3b21', 'z3y21', 'z1b22', 'z1y22',
#  'z2b22', 'z2y22', 'z3b22', 'z3y22', 'z1b23', 'z1y23',
#  'z2b23', 'z2y23', 'z3b23', 'z3y23', 'z1b24', 'z1y24',
#  'z2b24', 'z2y24', 'z3b24', 'z3y24', 'z1b25', 'z1y25',
#  'z2b25', 'z2y25', 'z3b25', 'z3y25']
