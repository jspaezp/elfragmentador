import torch
import torch.nn as nn
from ms_transformer_layers import (
    _IRTDecoder,
    _PeptideTransformerDecoder,
    _PeptideTransformerEncoder,
)
from torch import Tensor

from elfragmentador.named_batches import PredictionResults


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        """
        MLP implements a very simple multi-layer perceptron (also called FFN).

        Concatenates hidden linear layers with activations for n layers.
        This implementation uses gelu instead of relu
        (linear > gelu) * (n-1) > linear

        Based on: https://github.com/facebookresearch/detr/blob/models/detr.py#L289

        Parameters:
            input_dim (int):
                Expected dimensions for the input
            hidden_dim (int):
                Number of dimensions of the hidden layers
            output_dim (int):
                Output dimensions
            num_layers (int):
                Number of layers (total)
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass over the network.

        Args:
          x (Tensor):

        Returns:
            Tensor

        Examples:
            >>> pl.seed_everything(42)
            42
            >>> net = MLP(1000, 512, 2, 10)
            >>> out = net.forward(torch.rand([5, 1000]))
            >>> out
            tensor([[-0.0061, -0.0219],
                    [-0.0061, -0.0219],
                    [-0.0061, -0.0220],
                    [-0.0061, -0.0220],
                    [-0.0061, -0.0219]], grad_fn=<AddmmBackward0>)
            >>> out.shape
            torch.Size([5, 2])
        """
        for i, layer in enumerate(self.layers):
            x = (
                torch.nn.functional.gelu(layer(x))
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


class PeptransformerBase(nn.Module):
    def __init__(
        self,
        num_fragments,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhid: int = 2024,
        d_model: int = 516,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Peptide encoder
        self.encoder = _PeptideTransformerEncoder(
            d_model=d_model,
            dropout=dropout,
            nhead=nhead,
            nhid=nhid,
            layers=num_encoder_layers,
        )

        # Peptide decoder
        self.decoder = _PeptideTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            nhid=nhid,
            layers=num_decoder_layers,
            dropout=dropout,
            num_fragments=num_fragments,
        )

        self.irt_decoder = _IRTDecoder(
            d_model=d_model,
            dim_feedforward=nhid,
            nhead=nhead,
            n_layers=num_encoder_layers,
            dropout=dropout,
        )

    def forward(
        self,
        seq: Tensor,
        mods: Tensor,
        charge: Tensor,
        nce: Tensor,
    ) -> PredictionResults:
        """
        Forward Generate predictions.

        Privides the function for the forward pass to the model.

        Parameters:
            seq (Tensor): Encoded pepide sequence [B, L] (view details)
            mods (Tensor): Encoded modification sequence [B, L], by default None
            nce (Tensor): float Tensor with the charges [B, 1]
            charge (Tensor): long Tensor with the charges [B, 1], by default None

        Details:
            seq:
                The peptide is encoded as integers for the aminoacid.
                "AAA" encoded for a max length of 5 would be
                torch.Tensor([ 1,  1,  1,  0,  0]).long()
            nce:
                Normalized collision energy to use during the prediction.
            charge:
                A tensor corresponding to the charges of each of the
                peptide precursors (long)
            mods:
                Modifications encoded as integers
        """

        trans_encoder_output, mem_mask = self.encoder(seq=seq, mods=mods)

        rt_output = self.irt_decoder(seq=seq, mods=mods)

        spectra_output = self.decoder(
            memory=trans_encoder_output,
            charge=charge,
            nce=nce,
            memory_key_padding_mask=mem_mask,
        )

        return PredictionResults(irt=rt_output, spectra=spectra_output)
