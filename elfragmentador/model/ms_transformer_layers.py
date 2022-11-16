import math

import torch
import torch.nn as nn
from torch import Tensor

from elfragmentador.config import get_default_config
from elfragmentador.model.nn_encoding import AASequenceEmbedding, ConcatenationEncoder
from elfragmentador.model.transformer_layers import _LearnableEmbedTransformerDecoder

CONFIG = get_default_config()


class IRTDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=224,
        nhead=4,
        n_layers=3,
        dropout=0.05,
        final_decoder="linear",
    ):
        super().__init__()
        """Decode iRTs.

        It is technically an encoder-decoder...

        Args:
            d_model (int):
                Number of dimensions to expect as input
            nhead (int):
                Number of heads in the attention layers that decode the input.
                defaults to 4
            dim_feedforward (int, optional):
                Number of hidden dimensions in the FFN that decodes the sequence.
                Defaults to 224
            n_layers (int, optional):
                dropout to use in the multihead attention.
                Defaults to 3
        """

        self.aa_embed = AASequenceEmbedding(
            d_model=d_model,
            aa_names=CONFIG.encoding_aa_order,
            mod_names=CONFIG.encoding_mod_order,
            max_length=100,
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=n_layers
        )
        self.decoder = _LearnableEmbedTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            nhid=dim_feedforward,
            layers=n_layers,
            dropout=dropout,
            num_outputs=1,
            final_decoder=final_decoder,
        )

    def forward(self, seq, mods):
        # seq [N, S], mods [N, S]
        trans_encoder_mask = torch.zeros_like(seq, dtype=torch.float)
        trans_encoder_mask = trans_encoder_mask.masked_fill(
            seq <= 0, float("-inf")
        ).masked_fill(seq > 0, float(0.0))
        # mask [N, S]

        embed_seq = self.aa_embed(seq=seq, mods=mods)  # [S, N, d_model]

        memory = self.encoder(embed_seq, src_key_padding_mask=trans_encoder_mask)
        out = self.decoder(memory, trans_encoder_mask)
        return out


class PeptideTransformerEncoder(torch.nn.Module):
    def __init__(
        self, d_model: int, dropout: float, nhead: int, nhid: int, layers: int
    ) -> None:
        super().__init__()

        # Aminoacid embedding
        self.aa_embed = AASequenceEmbedding(
            d_model=d_model,
            aa_names=CONFIG.encoding_aa_order,
            mod_names=CONFIG.encoding_mod_order,
            max_length=100,
        )

        # Transformer encoder sections
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, layers)

    def forward(self, seq: Tensor, mods: Tensor) -> Tensor:
        # For the mask ....
        # If a BoolTensor is provided, positions with True are not allowed
        # to attend while False values will be unchanged <- form the pytorch docs

        # [1,1,0]
        # bool [True, True, False]
        # ~    [False, False, True]
        # [Attend, Attend, Dont Attend]

        # seq shape [N, S]
        trans_encoder_mask = torch.zeros_like(seq, dtype=torch.float)
        trans_encoder_mask = trans_encoder_mask.masked_fill(
            seq <= 0, float("-inf")
        ).masked_fill(seq > 0, float(0.0))

        x = self.aa_embed(seq=seq, mods=mods)
        # x shape [S, N, d_model]

        trans_encoder_output = self.encoder(x, src_key_padding_mask=trans_encoder_mask)
        # trans_encoder_output shape [S, N, d_model]

        return trans_encoder_output, trans_encoder_mask


class FragmentTransformerDecoder(_LearnableEmbedTransformerDecoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        nhid: int,
        layers: int,
        dropout: float,
        num_fragments: int,
        charge_dims_pct: float = 0.05,
        nce_dims_pct: float = 0.05,
        final_decoder: str = "linear",
    ) -> None:
        charge_dims = math.ceil(d_model * charge_dims_pct)
        nce_dims = math.ceil(d_model * nce_dims_pct)
        n_embeds = d_model - (charge_dims + nce_dims)

        super().__init__(
            d_model=d_model,
            embed_dims=n_embeds,
            nhead=nhead,
            nhid=nhid,
            layers=layers,
            dropout=dropout,
            num_outputs=num_fragments,
            final_decoder=final_decoder,
        )

        self.charge_encoder = ConcatenationEncoder(
            dims_add=charge_dims, max_val=10.0, scaling=math.sqrt(d_model)
        )
        self.nce_encoder = ConcatenationEncoder(
            dims_add=nce_dims, max_val=100.0, scaling=math.sqrt(d_model)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.trans_decoder_embedding.weight, -initrange, initrange)

    def preprocess_query(self, query, charge, nce):
        # [T, B, E2]
        trans_decoder_tgt = self.charge_encoder(query, charge)
        # [T, B, E1]
        trans_decoder_tgt = self.nce_encoder(trans_decoder_tgt, nce)
        # [T, B, E]
        return trans_decoder_tgt

    def forward(
        self,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        charge: Tensor,
        nce: Tensor,
    ) -> Tensor:
        trans_decoder_tgt = self.get_learnable_query(batch_size=charge.size(0))
        trans_decoder_tgt = self.preprocess_query(
            trans_decoder_tgt, nce=nce, charge=charge
        )
        # [T, B, E]

        output = self.decoder_forward(
            trans_decoder_tgt=trans_decoder_tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output
