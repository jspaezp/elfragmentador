try:
    from typing import Dict, List, Tuple, Optional, Union, Literal

    LiteralFalse = Literal[False]
except ImportError:
    # Python pre-3.8 compatibility
    from typing import Dict, List, Tuple, Optional, Union, NewType

    LiteralFalse = NewType("LiteralFalse", bool)

import math
import torch
from torch import Tensor, nn
from elfragmentador import constants
import pytorch_lightning as pl

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

    def forward(self, x: torch.LongTensor):
        """forward Concatenates the values to the 

        [extended_summary]

        Parameters
        ----------
        x : Tensor
            Tensor of shape [BatchSize, SequenceLength], this should encode
            a sequence and be padded with zeros

        Returns
        -------
        Tensor
            Tensor of shape [SequenceLength, BatchSize, DimensionsAdded],
            Where 
        """
        vals = x.bool().long()
        if self.inverted:
            vals = vals.flip(1)
            
        out = self.pe[vals.cumsum(1)]
        if self.inverted:
            out = out.flip(1)
            
        return out.transpose(1,0)


def test_inverted_positional_encoder():
    encoder = SeqPositionalEmbed(6, 50, inverted=True)
    x = torch.cat([torch.ones(1,2), torch.ones(1,2)*2, torch.zeros((1,2))], dim = -1).long()
    x[0]
    # tensor([1, 1, 2, 2, 0, 0])
    x.shape
    # torch.Size([1, 6])
    out = encoder(x)
    assert out.shape == torch.Size((6,1,6))
    assert torch.all(out[5,0,:] == 0)
    assert torch.all(out[4,0,:] == 0)
    assert torch.all(out[3,0,:] != 0)
    assert torch.all(out[3,0,:] == encoder.pe[1])
    assert torch.any(out[3,0,:] == encoder.pe[2])
    assert torch.any(out[3,0,:] != out[0,0,:])

    encoder = SeqPositionalEmbed(6, 50, inverted=True)
    input_t = torch.tril(torch.ones((3,3))).long()
    input_t[1]
    # tensor([1, 1, 0])
    out = encoder(input_t)
    assert out.shape == torch.Size((3,3,6))
    assert torch.all(out[2, 1, :] == 0)
    assert torch.any(out[1, 1, :] != 0)
    assert torch.any(out[0, 1, :] != 0)
    assert torch.any(out[0, 1, :] != out[1, 1, :])


class ConcatenationEncoder(torch.nn.Module):
    """ConcatenationEncoder concatenates information into the embedding.

    Adds information on continuous variables into an embedding by concatenating an n number
    of dimensions to it

    Parameters
    ----------
    dims_add : int
        Number of dimensions to add as an encoding
    dropout : float, optional
        dropout, by default 0.1
    max_val : float, optional
        maximum expected value of the variable that will be encoded, by default 200.0
    static_size : Union[Literal[False], float], optional
        Optional ingeter to pass in order to make the size deterministic.
        This is only required if you want to export your model to torchscript, by default False

    Examples
    --------
    >>> x1 = torch.zeros((5, 1, 20))
    >>> x2 = torch.zeros((5, 2, 20))
    >>> encoder = ConcatenationEncoder(10, 0.1, 10)
    >>> output = encoder(x1, torch.tensor([[7]]))
    >>> output = encoder(x2, torch.tensor([[7], [4]]))
    """
    # TODO evaluate if fropout is actually useful here ...

    def __init__(
        self,
        dims_add: int,
        dropout: float = 0.1,
        max_val: Union[float, int] = 200.0,
        static_size: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # pos would be a variable ...
        div_term = torch.exp(
            torch.arange(0, dims_add, 2).float()
            * (-math.log(float(2 * max_val)) / (dims_add))
        )
        self.register_buffer("div_term", div_term)
        self.static_size = static_size
        self.dims_add = dims_add

    def forward(self, x: Tensor, val: Tensor, debug: bool = False) -> Tensor:
        r"""Forward pass thought the encoder.

        Args
        ----
        x:
            the sequence fed to the encoder model (required).
        val:
            value to be encoded into the sequence (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            val: [batch size, 1]
            output: [sequence length, batch size, embed_dim + added_dims]

        Examples
        --------
        >>> x1 = torch.zeros((5, 1, 20))
        >>> x2 = torch.cat([x1, x1+1], axis = 1)
        >>> encoder = ConcatenationEncoder(10, dropout = 0, max_val = 10)
        >>> output = encoder(x1, torch.tensor([[7]]))
        >>> output.shape
        torch.Size([5, 1, 30])
        >>> output = encoder(x2, torch.tensor([[7], [4]]))
        """
        if debug:
            print(f"CE: Shape of inputs val={val.shape} x={x.shape}")

        if self.static_size:
            assert self.static_size == x.size(0), (
                f"Size of the first dimension ({x.size(0)}) "
                f"does not match the expected value ({self.static_size})"
            )
            end_position = self.static_size
        else:
            end_position = x.size(0)

        e_sin = torch.sin(val * self.div_term)
        e_cos = torch.cos(torch.cos(val * self.div_term))
        e = torch.cat([e_sin, e_cos], axis=-1)

        if debug:
            print(f"CE: Making encodings e={e.shape}")

        assert e.shape[-1] < self.dims_add + 2, (
            "Internal error in concatenation encoder"
        )
        e = e[...,:self.dims_add]

        if debug:
            print(f"CE: clipping encodings e={e.shape}")

        e = torch.cat([e.unsqueeze(0)] * end_position)

        if debug:
            print(f"CE: Shape before concat e={e.shape} x={x.shape}")

        x = torch.cat((x, e), axis=-1)
        if debug:
            print(f"CE: Shape after concat x={x.shape}")
        return self.dropout(x)


class PositionalEncoding(torch.nn.Module):
    r"""PositionalEncoding adds positional information to tensors.

    Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    Args
    ----
    d_model: int
       the embed dim (required), must be even.
    dropout: float
       the dropout value (default=0.1).
    max_len: int
       the max. length of the incoming sequence (default=5000).
    static_size : Union[LiteralFalse, int], optional
        If it is an integer it is the size of the inputs that will
        be given, it is used only when tracing the model for torchscript
        (since torchscript needs fixed length inputs), by default False

    Examples
    --------
    >>> posencoder = PositionalEncoding(20, 0.1, max_len=20)
    >>> x = torch.ones((2,1,20)).float()
    >>> x.shape
    torch.Size([2, 1, 20])
    >>> posencoder(x).shape
    torch.Size([2, 1, 20])

    Therefore encoding are (seq_length, batch, encodings)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        static_size: Union[LiteralFalse, int] = False,
    ) -> None:
        """__init__ Creates a new instance."""
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        self.static_size = static_size

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass though the encoder.

        Args
        ----
            x: the sequence fed to the positional encoder model (required).
        Shape
        -----
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]

        Examples
        --------
        >>> pl.seed_everything(42)
        42
        >>> x = torch.ones((1,4,6)).float()
        >>> pos_encoder = PositionalEncoding(6, 0.1, max_len=10)
        >>> output = pos_encoder(x)
        >>> output.shape
        torch.Size([1, 4, 6])
        >>> output
        tensor([[[1.1111, 2.2222, 1.1111, 2.2222, 1.1111, 2.2222],
               [1.1111, 2.2222, 1.1111, 2.2222, 1.1111, 2.2222],
               [1.1111, 2.2222, 1.1111, 2.2222, 1.1111, 0.0000],
               [1.1111, 2.2222, 1.1111, 2.2222, 1.1111, 2.2222]]])
        """
        if self.static_size:
            end_position = self.static_size
        else:
            end_position = x.size(0)

        x = x + self.pe[:end_position, :]
        return self.dropout(x)


class AASequenceEmbedding(torch.nn.Module):
    def __init__(self, ninp, position_ratio=0.1):
        super().__init__()
        positional_ninp = int((ninp/2) * position_ratio)
        if positional_ninp % 2:
            positional_ninp += 1
        ninp_embed = int(ninp - (2*positional_ninp))

        # Positional information additions
        self.fw_position_embed = SeqPositionalEmbed(
            max_len=constants.MAX_SEQUENCE * 4,
            dims_add=positional_ninp,
            inverted=False)
        self.rev_position_embed = SeqPositionalEmbed(
            max_len=constants.MAX_SEQUENCE * 4,
            dims_add=positional_ninp,
            inverted=True)

        # Aminoacid embedding
        self.aa_encoder = nn.Embedding(
            constants.AAS_NUM + 1, ninp_embed, padding_idx=0
        )
        # PTM embedding
        self.mod_encoder = nn.Embedding(
            len(constants.MODIFICATION) + 1, ninp_embed, padding_idx=0
        )

        # Weight Initialization
        self.init_weights()
        self.ninp = ninp_embed

    def init_weights(self) -> None:
        initrange = 0.1
        ptm_initrange = initrange * 0.01
        torch.nn.init.uniform_(self.aa_encoder.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.mod_encoder.weight, -ptm_initrange, ptm_initrange)

    def forward(self, src, mods, debug: bool=False):
        if debug:
            print(f"AAE: Input shapes src={src.shape}, mods={mods.shape}")
        fw_pos_emb = self.fw_position_embed(src)
        rev_pos_emb = self.rev_position_embed(src)

        src = self.aa_encoder(src.permute(1, 0))
        mods = self.mod_encoder(mods.permute(1, 0))
        src = src + mods

        # TODO consider if this line is needed
        src = src * math.sqrt(self.ninp)
        if debug:
            print(f"AAE: Shape after embedding {src.shape}")

        src = torch.cat([src, fw_pos_emb, rev_pos_emb], dim = -1)
        if debug:
            print(f"AAE: Shape after embedding positions {src.shape}")

        return src