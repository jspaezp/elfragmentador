"""
Implements torch models to handle encoding and decoding of positions as well as
learnable embeddings for the aminoacids and ions.
"""

try:
    from typing import Literal, Union

    LiteralFalse = Literal[False]
except ImportError:
    # Python pre-3.8 compatibility
    from typing import NewType, Union

    LiteralFalse = NewType("LiteralFalse", bool)

import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from elfragmentador import constants


class SeqPositionalEmbed(torch.nn.Module):
    def __init__(self, dims_add: int = 10, max_len: int = 30, inverted=True):
        super().__init__()
        pe = torch.zeros(max_len, dims_add)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term_enum = torch.arange(0, dims_add, 2).float()
        div_term_denom = -math.log(10000.0) / dims_add + 1
        div_term = torch.exp(div_term_enum * div_term_denom)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe[0, :] = 0
        self.register_buffer("pe", pe)
        self.inverted = inverted

    def forward(self, x: torch.LongTensor) -> Tensor:
        """
        forward.

        Returns the positional encoding for a sequence, expressed as integers

        Parameters:
            x (Tensor):
                Integer Tensor of shape **[BatchSize, SequenceLength]**,
                this should encode a sequence and be padded with zeros

        Returns:
            Tensor (Tensor), The positional encoding ...

        Examples:
            >>> encoder = SeqPositionalEmbed(6, 50, inverted=True)
            >>> encoder2 = SeqPositionalEmbed(6, 50, inverted=False)
            >>> x = torch.cat([
            ... torch.ones(1,2),
            ... torch.ones(1,2)*2,
            ... torch.zeros((1,2))],
            ... dim = -1).long()
            >>> x[0]
            tensor([1, 1, 2, 2, 0, 0])
            >>> x.shape
            torch.Size([1, 6])
            >>> out = encoder(x)
            >>> out2 = encoder2(x)
            >>> out.shape
            torch.Size([6, 1, 6])
            >>> out
            tensor([[[-0.7568, -0.6536,  0.9803,  0.1976,  0.4533,  0.8913]],
                [[ 0.1411, -0.9900,  0.8567,  0.5158,  0.3456,  0.9384]],
                [[ 0.9093, -0.4161,  0.6334,  0.7738,  0.2331,  0.9725]],
                [[ 0.8415,  0.5403,  0.3363,  0.9418,  0.1174,  0.9931]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]])
            >>> out2
            tensor([[[ 0.8415,  0.5403,  0.3363,  0.9418,  0.1174,  0.9931]],
                [[ 0.9093, -0.4161,  0.6334,  0.7738,  0.2331,  0.9725]],
                [[ 0.1411, -0.9900,  0.8567,  0.5158,  0.3456,  0.9384]],
                [[-0.7568, -0.6536,  0.9803,  0.1976,  0.4533,  0.8913]],
                [[-0.7568, -0.6536,  0.9803,  0.1976,  0.4533,  0.8913]],
                [[-0.7568, -0.6536,  0.9803,  0.1976,  0.4533,  0.8913]]])
        """
        vals = x.bool().long()
        if self.inverted:
            vals = vals.flip(1)

        out = self.pe[vals.cumsum(1)]
        if self.inverted:
            out = out.flip(1)

        return out.transpose(1, 0)


class ConcatenationEncoder(torch.nn.Module):
    # TODO evaluate if fropout is actually useful here ...
    def __init__(
        self,
        dims_add: int,
        max_val: Union[float, int] = 200.0,
        static_size: bool = False,
        scaling=1,
    ) -> None:
        r"""ConcatenationEncoder concatenates information into the embedding.

        Adds information on continuous variables into an embedding by concatenating
        an n number of dimensions to it.

        It is meant to add different information to every element in a batch, but the
        same information (number of dimensions) to every element of a sequence inside
        an element of the batch. \(x[i_1,j,-y:] = x[i_2,j,-y:]\) ; being \(y\) the
        number of added dimensions.

        Args:
            dims_add (int): Number of dimensions to add as an encoding
            max_val (float, optional):
                maximum expected value of the variable that will be encoded,
                by default 200.0
            static_size (Union[Literal[False], float], optional):
                Optional ingeter to pass in order to make the size deterministic.
                This is only required if you want to export your model to torchscript,
                by default False

        Examples:
            >>> x1 = torch.zeros((5, 1, 20))
            >>> x2 = torch.zeros((5, 2, 20))
            >>> encoder = ConcatenationEncoder(dims_add = 10, max_val=10)
            >>> output = encoder(x1, torch.tensor([[7]]))
            >>> output = encoder(x2, torch.tensor([[7], [4]]))
        """
        super().__init__()

        # pos would be a variable ...
        div_term = torch.exp(
            torch.arange(0, dims_add, 2).float()
            * (-math.log(float(2 * max_val)) / (dims_add))
        )
        self.register_buffer("div_term", div_term)
        # TODO add option to make trainable
        self.static_size = static_size
        self.dims_add = dims_add
        self.scaling = scaling

    def forward(self, x: Tensor, val: Tensor) -> Tensor:
        """
        Forward pass thought the encoder.

        Parameters:
            x (Tensor):
                the sequence fed to the encoder model (required).
                shape is **[sequence length, batch size, embed dim]**.
            val (Tensor):
                value to be encoded into the sequence (required).
                Shape is **[batch size, 1]**.

        Returns:
            Tensor (Tensor),
            Of shape **[sequence length, batch size, embed_dim + added_dims]**

        Examples:
            >>> x1 = torch.zeros((5, 1, 20))
            >>> x2 = torch.cat([x1, x1+1], axis = 1)
            >>> encoder = ConcatenationEncoder(10, max_val = 10)
            >>> output = encoder(x1, torch.tensor([[7]]))
            >>> output.shape
            torch.Size([5, 1, 30])
            >>> output = encoder(x2, torch.tensor([[7], [4]]))
        """

        e_sin = torch.sin(val * self.div_term)
        e_cos = torch.cos(val * self.div_term)
        e = torch.cat([e_sin, e_cos], axis=-1)

        assert (
            e.shape[-1] < self.dims_add + 2
        ), "Internal error in concatenation encoder"

        e = e[..., : self.dims_add]
        e = e.unsqueeze(0).expand(x.size(0), -1, -1) / self.scaling
        x = torch.cat((x, e), axis=-1)

        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        static_size: Union[LiteralFalse, int] = False,
    ) -> None:
        r"""PositionalEncoding adds positional information to tensors.

        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.

        \({PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model)\)
        \({PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))\)

        where pos is the word position and i is the embed idx)

        Args:
            d_model (int):
                the embed dim (required), must be even.
            max_len (int):
                the max. length of the incoming sequence (default=5000).
            static_size (Union[LiteralFalse, int], optional):
                If it is an integer it is the size of the inputs that will
                be given, it is used only when tracing the model for torchscript
                (since torchscript needs fixed length inputs), by default False

        Note:
            Therefore encoding are **(seq_length, batch, encodings)**

        Examples:
            >>> posencoder = PositionalEncoding(20, max_len=20)
            >>> x = torch.ones((2,1,20)).float()
            >>> x.shape
            torch.Size([2, 1, 20])
            >>> posencoder(x).shape
            torch.Size([2, 1, 20])
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) / math.sqrt(d_model)
        # Pe has [shape max_len, 1, d_model]
        self.register_buffer("pe", pe)
        self.static_size = static_size

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass though the encoder.

        Args:
            x (Tensor):
                the sequence fed to the positional encoder model (required).
                Shape **[sequence length, batch size, embed dim]**

        Returns:
            Tensor (Tensor), of shape **[sequence length, batch size, embed dim]**

        Examples:
            >>> import pytorch_lightning as pl
            >>> pl.seed_everything(42)
            42
            >>> x = torch.ones((4,1,6)).float()
            >>> pos_encoder = PositionalEncoding(6, max_len=10)
            >>> output = pos_encoder(x)
            >>> output.shape
            torch.Size([4, 1, 6])
            >>> output
            tensor([[[...]],
                [[...]],
                [[...]],
                [[...]]])
        """
        if self.static_size:
            end_position = self.static_size
        else:
            end_position = x.size(0)

        x = x + self.pe[:end_position, :]
        return x


class AASequenceEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Positional information additions
        self.position_embed = PositionalEncoding(
            d_model=d_model,
            max_len=constants.MAX_TENSOR_SEQUENCE * 4,
        )

        # Aminoacid embedding
        self.aa_encoder = nn.Embedding(constants.AAS_NUM + 1, d_model, padding_idx=0)
        # PTM embedding
        self.mod_encoder = nn.Embedding(
            len(constants.MODIFICATION) + 1, d_model, padding_idx=0
        )

        # Weight Initialization
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        ptm_initrange = initrange * 0.01
        torch.nn.init.uniform_(self.aa_encoder.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.mod_encoder.weight, -ptm_initrange, ptm_initrange)

    def forward(self, seq, mods):
        # seq and mod are [N, S] shaped
        mods = F.pad(mods, (0, seq.size(1) - mods.size(1)), "constant")
        seq = self.aa_encoder(seq.permute(1, 0))
        mods = self.mod_encoder(mods.permute(1, 0))
        seq = seq + mods

        # TODO consider if this line is needed, it is used in attention is all you need
        seq = seq * math.sqrt(self.aa_encoder.num_embeddings)
        seq = self.position_embed(seq)

        return seq

    def as_DataFrames(self):
        """
        Returns the weights as data frames.

        Returns:
            Tuple[DataFrame, DataFrame]:
                A data frame of the aminoacid embeddings and the modification embeddings

        Examples:
            >>> embed = AASequenceEmbedding(20)
            >>> aa_embed, mod_embed = embed.as_DataFrames()
            >>> list(aa_embed)
            ['EMPTY', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',\
                'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'c', 'n']
            >>> list(mod_embed)
            ['EMPTY', 'CARBAMIDOMETHYL', 'ACETYL', 'DEAMIDATED', 'OXIDATION', \
                'PHOSPHO', 'METHYL', 'DIMETHYL', 'TRIMETHYL', 'FORMYL', 'GG', \
                'LRGG', 'NITRO', 'BIOTINYL', 'TMT6PLEX']
        """
        df_aa = pd.DataFrame(data=self.aa_encoder.weight.detach().numpy().T)
        df_aa.columns = ["EMPTY"] + list(constants.ALPHABET.keys())

        df_mod = pd.DataFrame(data=self.mod_encoder.weight.detach().cpu().numpy().T)
        df_mod.columns = ["EMPTY"] + list(constants.MODIFICATION)

        return df_aa, df_mod
