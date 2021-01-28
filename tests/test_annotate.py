from elfragmentador import annotate


def test_peptide_parser():
    print(list(annotate.peptide_parser("AAACC")))
    print(list(annotate.peptide_parser("AAA[+53]CC")))
    print(list(annotate.peptide_parser("AAA[+53]CC[+54]")))
    print(list(annotate.peptide_parser("__AAA[+53]CC[+54]__")))
    print(list(annotate.peptide_parser("__AAA[53]CC[54]__")))
