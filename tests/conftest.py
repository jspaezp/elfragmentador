import pytest
from elfragmentador.model import PepTransformerModel

# Arrange
@pytest.fixture(scope="session")
def tiny_model():
    return PepTransformerModel(
        num_decoder_layers=3, num_encoder_layers=2, nhid=64, ninp=64, nhead=2
    )
