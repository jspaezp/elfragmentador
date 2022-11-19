from elfragmentador.data import datamodules
from elfragmentador.model import PepTransformerModel


def tiny_model_builder():
    mod = PepTransformerModel(
        num_decoder_layers=3,
        num_encoder_layers=2,
        nhid=48,
        d_model=48,
        nhead=2,
        dropout=0,
        lr=1e-4,
        scheduler="cosine",
        loss_ratio=1000,
        lr_ratio=10,
    )
    mod.eval()
    return mod


def datamodule_builder(shared_datadir):
    dm = datamodules.TrainingDataModule(
        batch_size=4,
        base_dir=str(shared_datadir / "parquet"),
    )
    return dm
