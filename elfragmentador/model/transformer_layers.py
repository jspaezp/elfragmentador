from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor


class _LearnableEmbedTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        nhid: int,
        layers: int,
        dropout: float,
        num_outputs: int,
        embed_dims: Optional[int] = None,
    ) -> None:
        """Implements a transformer decoder with a learnable embedding layer."""
        super().__init__()
        logger.info(
            "Creating TransformerDecoder"
            f" nhid={nhid}, "
            f"d_model={d_model} "
            f"nhead={nhead} "
            f"layers={layers}"
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation="gelu",
        )
        self.trans_decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        # self.peak_decoder = MLP(d_model, d_model, output_dim=1, num_layers=2)
        self.peak_decoder = nn.Linear(d_model, 1)

        logger.info(f"Creating embedding for spectra of length {num_outputs}")
        self.trans_decoder_embedding = nn.Embedding(
            num_embeddings=num_outputs,
            embedding_dim=d_model if embed_dims is None else embed_dims,
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.trans_decoder_embedding.weight, -initrange, initrange)

    def get_learnable_query(self, batch_size):
        trans_decoder_tgt = self.trans_decoder_embedding.weight.unsqueeze(1)
        # [T, E2] > [T, 1, E2]
        trans_decoder_tgt = trans_decoder_tgt.expand(-1, batch_size, -1)
        return trans_decoder_tgt

    def preprocess_query(self, query):
        """Preprrcess the query before passing it to the decoder.

        This is used by self.forward to preprocess the query before
        passing it to the decoder in self.forward. Subclasses of this
        class can override this method to implement custom preprocessing.
        """
        return query

    def decoder_forward(
        self, trans_decoder_tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor
    ):
        spectra_output = self.trans_decoder(
            memory=memory,
            tgt=trans_decoder_tgt,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        # Shape is [NumFragments, Batch, NumEmbed]

        spectra_output = self.peak_decoder(spectra_output)
        # Shape is [NumFragments, Batch, 1]
        spectra_output = spectra_output.squeeze(-1).permute(1, 0)
        # Shape is [Batch, NumFragments]
        return spectra_output

    def forward(
        self,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        """Has to be implemente when subclassing."""
        trans_decoder_tgt = self.get_learnable_query(batch_size=memory.size(1))
        trans_decoder_tgt = self.preprocess_query(trans_decoder_tgt)

        output = self.decoder_forward(
            trans_decoder_tgt=trans_decoder_tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output
