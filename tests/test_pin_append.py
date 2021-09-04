from elfragmentador.datasets.percolator import append_preds


def test_append_predictions(shared_datadir, tmpdir, tiny_model):
    input_pin = "01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.pin"
    input_pin = shared_datadir / input_pin
    out = append_preds(in_pin=input_pin, out_pin=tmpdir / "out.pin", model=tiny_model)
    assert "SpecAngle" in list(out)
