from elfragmentador.model import PepTransformerModel


def tiny_model_builder():
    mod = PepTransformerModel(
        num_decoder_layers=3,
        num_encoder_layers=2,
        nhid=112,
        d_model=112,
        nhead=2,
        dropout=0,
        lr=1e-4,
        scheduler="cosine",
        loss_ratio=1000,
        lr_ratio=10,
    )
    mod.eval()
    return mod
