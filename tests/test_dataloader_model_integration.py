from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader

from elfragmentador import datamodules, model


def test_dataset_input_works_on_model(shared_datadir):
    mod = model.PepTransformerModel(nhead=4, d_model=64)
    mod.eval()
    ds = datamodules.PeptideDataset.from_sptxt(str(shared_datadir) + "/sample.sptxt")
    dl = DataLoader(ds, 4)
    inputs = next(dl.__iter__())

    mod.forward(seq=inputs.seq, mods=inputs.mods, charge=inputs.charge, nce=inputs.nce)

    with torch.no_grad():
        for i, inputs in enumerate(dl):
            print(f"Batch Number: {i}")
            mod(seq=inputs.seq, mods=inputs.mods, charge=inputs.charge, nce=inputs.nce)
