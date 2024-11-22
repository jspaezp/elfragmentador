import pytorch_lightning as pl
from flaky import flaky

from elfragmentador.data.predictor import Predictor
from elfragmentador.model import PepTransformerModel

pl.seed_everything(420)


@flaky(max_runs=6, min_passes=2)
def test_prediction_loop(shared_datadir, tmpdir):
    model = PepTransformerModel(d_model=96)
    model = model.eval()
    Predictor(model=model)
    # predictor = Predictor(model=model)
    # fasta_file = shared_datadir / "fasta/P0DTC4.fasta"
    # out_dlib = tmpdir / "out.dlib"
    # predictor.predict_to_file(fasta_file, out_dlib, nce=30, charges=[2])
    # predictor.compare(str(out_dlib), nce=30)
