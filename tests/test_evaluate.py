from pathlib import Path
import elfragmentador as ef
from elfragmentador import model, datamodules, evaluate
from elfragmentador import DEFAULT_CHECKPOINT


def test_evaluation_on_dataset_works(shared_datadir):
    mod = model.PepTransformerModel.load_from_checkpoint(ef.DEFAULT_CHECKPOINT)
    mod.eval()
    ds = datamodules.PeptideDataset.from_sptxt(
        str(shared_datadir) + "/small_phospho_spectrast.sptxt"
    )
    out = evaluate.evaluate_on_dataset(mod, ds)
    print(out)


def test_irt_evaluation_works():
    mod = model.PepTransformerModel.load_from_checkpoint(DEFAULT_CHECKPOINT)
    evaluate.evaluate_landmark_rt(model=mod)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    test_evaluation_on_dataset_works(str(parent_dir) + "/data/")
    test_irt_evaluation_works()
