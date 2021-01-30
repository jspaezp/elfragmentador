from pathlib import Path
from elfragmentador import model, datamodules, evaluate


def test_evaluation_on_dataset_works(shared_datadir):
    mod = model.PepTransformerModel(nhead=4, ninp=64)
    mod.eval()
    ds = datamodules.PeptideDataset.from_sptxt(
        str(shared_datadir) + "/small_phospho_spectrast.sptxt"
    )
    out = evaluate.evaluate_on_dataset(mod, ds)
    print(out)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    test_evaluation_on_dataset_works(str(parent_dir) + "/data/")
