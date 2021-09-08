import pytorch_lightning as pl
from elfragmentador.predictor import Predictor


def test_prediction_loop(datamodule, tiny_model):
    trainer = Predictor()
    out = trainer.predict(tiny_model, datamodule.train_dataloader())

    assert isinstance(out, tuple)
    assert hasattr(out, "_fields")

    assert "irt" in out._fields and "spectra" in out._fields

    assert len(out) == 2
    assert len(set([len(x) for x in out])) == 1


def test_testing_loop(datamodule, tiny_model):
    trainer = Predictor()
    out = trainer.test(
        tiny_model, datamodule.train_dataloader(), ckpt_path=None, plot=False
    )

    assert isinstance(out, tuple)
    assert hasattr(out, "_fields")
    assert len(out) == 4

    expected_outs = ["scaled_se_loss", "loss_cosine", "loss_irt", "loss_angle"]
    assert all([x in out._fields for x in expected_outs])
