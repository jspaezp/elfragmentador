from elfragmentador import DEFAULT_CHECKPOINT
from elfragmentador.model import PepTransformerModel
from elfragmentador.utils import append_preds


def test_append_predictions(shared_datadir, tmpdir):
    input_pin = "01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.pin"
    input_pin = shared_datadir / input_pin
    model = PepTransformerModel.load_from_checkpoint(DEFAULT_CHECKPOINT)
    out = append_preds(in_pin=input_pin, out_pin=tmpdir / "out.pin", model=model)
    assert "SpecCorrelation" in list(out)
