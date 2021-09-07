from pathlib import Path
import elfragmentador as ef
from elfragmentador import datamodules, evaluate
from elfragmentador.predictor import Predictor


def test_evaluation_on_dataset_works(shared_datadir, tiny_model):
    mod = tiny_model
    mod.eval()
    ds = datamodules.PeptideDataset.from_sptxt(
        str(shared_datadir) + "/small_phospho_spectrast.sptxt"
    )
    predictor = Predictor()
    out = predictor.evaluate_dataset(mod, ds, optimize_nce=False)
    print(out)


def test_irt_evaluation_works(tiny_model):
    evaluate.evaluate_landmark_rt(model=tiny_model)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    test_evaluation_on_dataset_works(str(parent_dir) + "/data/")
    test_irt_evaluation_works()
