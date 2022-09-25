from elfragmentador import datamodules
from elfragmentador.datasets import Predictor


def test_evaluation_on_dataset_works(shared_datadir, tiny_model):
    mod = tiny_model
    mod.eval()
    ds = datamodules.PeptideDataset.from_sptxt(
        str(shared_datadir) + "/small_phospho_spectrast.sptxt"
    )
    predictor = Predictor()
    out = predictor.evaluate_dataset(mod, ds, optimize_nce=False)
    print(out)

    assert isinstance(out, tuple) and hasattr(out, "_fields")
    expect_fields = ["scaled_se_loss", "loss_cosine", "loss_irt", "loss_angle"]
    assert all([x in expect_fields for x in out._fields])
