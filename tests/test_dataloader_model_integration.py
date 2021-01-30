from pathlib import Path
import torch
from elfragmentador import model, datamodules
from torch.utils.data.dataloader import DataLoader


def test_dataset_input_works_on_model(shared_datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    mod.eval()
    ds = datamodules.PeptideDataset.from_sptxt(str(shared_datadir) + "/sample.sptxt")
    dl = DataLoader(ds, 4)
    inputs = ds[0]

    mod.batch_forward(inputs, debug=True)

    with torch.no_grad():
        for i, b in enumerate(dl):
            print(f"Batch Number: {i}")
            mod.batch_forward(b, debug=True)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    test_dataset_input_works_on_model(str(parent_dir) + "/data/")
