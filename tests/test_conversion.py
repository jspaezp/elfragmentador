import sys

import numpy as np
from loguru import logger
from ms2ml.data.adapters import MSPAdapter

from elfragmentador.config import CONFIG
from elfragmentador.data.converter import DeTensorizer, Tensorizer

logger.remove()
logger.add(sys.stderr, level="INFO")

speclist = list(
    MSPAdapter(
        config=CONFIG,
        file="/Users/sebastianpaez/Downloads/FTMS_HCD_20_annotated_2019-11-12_filtered.msp",
    ).parse()
)


def back_and_forth(spec):
    (
        seq,
        mods,
        charge,
        nce,
        spectra_tensor,
        irt,
        weight,
    ) = Tensorizer.convert_annotated_spectrum(spec, 27)
    spec = DeTensorizer.make_spectrum(
        seq=seq, mod=mods, charge=charge, fragment_vector=spectra_tensor, irt=irt
    )
    return spec


for spec in speclist:
    spec = speclist[0].normalize_intensity()
    round1 = back_and_forth(spec)
    round2 = back_and_forth(round1)

    assert np.allclose(round1.mz, round2.mz)
    assert len(round1.fragment_intensities) > 5

    all_ints = np.array(
        [(spec[x], round1[x], round2[x]) for x in round1.fragment_intensities]
    )
    all_ints = all_ints / all_ints.max(axis=0)

    assert np.allclose(
        np.min(all_ints, axis=1), np.max(all_ints, axis=1)
    ), spec.precursor_peptide.to_proforma()
