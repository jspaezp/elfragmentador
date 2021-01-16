import pandas as pd
from transprosit import constants, annotate


def encode_mod_seq(seq):
    """
    Encodes a peptide sequence to a numeric vector

    Example
    =======
    >>> samp_seq = "_AAIFVVAR_"
    >>> print(constants.MAX_SEQUENCE)
    30
    >>> out = encode(samp_seq)
    >>> out
    [1, 1, 8, 5, 18, 18, 1, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> len(out)
    30
    """
    max_len = constants.MAX_SEQUENCE
    AAS = "".join(constants.ALPHABET)
    seq = seq.replace("_", "")
    out = [0] * max_len

    try:
        out_i = [AAS.index(x) + 1 for x in seq]
        out[: len(out_i)] = out_i
    except ValueError:
        print(seq)
        raise ValueError
    return out


def decode_mod_seq(encoding):
    out = []

    for i in encoding:
        if i == 0:
            break

        out.append(constants.ALPHABET_S[i])

    return "".join(out)


# TODO add PTM parsing


def get_fragment_encoding_labels(annotated_peaks=None):
    """
    Gets either the laels or an sequence that encodes a spectra

    Examples
    ========
    >>> get_fragment_encoding_labels()
    ['z1b1', 'z1y1', 'z2b1', 'z2y1', 'z1b2', 'z1y2', 'z2b2', 'z2y2']

    >>> get_fragment_encoding_labels({'z1y2': 100, 'z2y2': 52})
    [0, 0, 0, 0, 0, 100, 0, 52]
    """

    encoding = []
    ion_encoding_iterables = {
        "ION_TYPE": "".join(sorted(constants.ION_TYPES)),
        "CHARGE": [f"z{z}" for z in range(1, constants.MAX_FRAG_CHARGE + 1)],
        "POSITION": list(range(1, constants.MAX_ION + 1)),
    }

    # TODO implement neutral losses ...  if needed
    for charge in ion_encoding_iterables[constants.ION_ENCODING_NESTING[0]]:
        for pos in ion_encoding_iterables[constants.ION_ENCODING_NESTING[1]]:
            for ion in ion_encoding_iterables[constants.ION_ENCODING_NESTING[2]]:
                key = f"{charge}{ion}{pos}"
                if annotated_peaks is None:
                    encoding.append(key)
                else:
                    encoding.append(annotated_peaks.get(key, 0))

    return encoding


def decode_fragment_tensor(
    sequence,
    tensor,
):
    """
    Returns a data frame with annotations from sequence
    and a tensor encoding a spectra

    Example
    =======
    >>> foo = decode_tensor("AAACK", torch.rand((150)), 3, "by", 25)
    >>> >>> foo.head()
      Fragment        Mass  Intensity
    0     z1b1   72.044390   0.887933
    1     z1y1  147.112804   0.765287
    2     z2b1   36.525833   0.420042
    3     z2y1   74.060040   0.428646
    4     z3b1   24.686314   0.212731
    >>> # import matplotlib.pyplot as plt
    >>> # plt.vlines(foo['Mass'], 0, foo['Intensity'])
    >>> # plt.show()
    """
    max_charge = constants.MAX_FRAG_CHARGE
    ions = "".join(sorted(constants.ION_TYPES))

    key_list = get_fragment_encoding_labels(annotated_peaks=None)
    fragment_ions = annotate.get_peptide_ions(
        sequence, list(range(1, max_charge + 1)), ion_types=ions
    )
    masses = [fragment_ions.get(k, 0) for k in key_list]
    intensities = [float(x) for x in tensor]

    assert len(intensities) == len(masses), print(
        f"Int {len(intensities)}: \n{intensities}\n\nmasses {len(masses)}: \n{masses}"
    )

    out_dict = {"Fragment": key_list, "Mass": masses, "Intensity": intensities}
    out_df = pd.DataFrame(out_dict)
    out_df = out_df[out_df["Mass"] != 0].copy()

    return out_df
