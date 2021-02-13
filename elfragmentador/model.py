try:
    from typing import Dict, List, Tuple, Optional, Union, Literal

    LiteralFalse = Literal[False]
except ImportError:
    # Python pre-3.8 compatibility
    from typing import Dict, List, Tuple, Optional, Union, NewType

    LiteralFalse = NewType("LiteralFalse", bool)

import warnings
import math
import time
from collections import namedtuple

import torch
from torch import Tensor, nn
import pytorch_lightning as pl

from argparse import _ArgumentGroup, ArgumentParser

import elfragmentador
from elfragmentador import constants
from elfragmentador import encoding_decoding
from elfragmentador.datamodules import TrainBatch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)

PredictionResults = namedtuple("PredictionResults", "irt spectra")


class MLP(nn.Module):
    """MLP implements a very simple multi-layer perceptron (also called FFN).

    Concatenates hidden linear layers with activations for n layers.
    This implementation uses gelu instead of relu
    (linear > gelu) * (n-1) > linear

    Based on: https://github.com/facebookresearch/detr/blob/models/detr.py#L289
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        """__init__ create a new instance of the MLP.

        Parameters
        ----------
        input_dim : int
            Expected dimensions for the input
        hidden_dim : int
            Number of dimensions of the hidden layers
        output_dim : int
            Output dimensions
        num_layers : int
            Number of layers (total)
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass over the network.

        Parameters
        ----------
        x : Tensor
            Dimensions should match the ones specified instantiating the class

        Returns
        -------
        Tensor
            The dims of this tensor are defined when instantiating the class

        Examples
        --------
        >>> pl.seed_everything(42)
        42
        >>> net = MLP(1000, 512, 2, 10)
        >>> out = net.forward(torch.rand([5, 1000]))
        >>> out
        tensor([[-0.0061, -0.0219],
                [-0.0061, -0.0219],
                [-0.0061, -0.0220],
                [-0.0061, -0.0220],
                [-0.0061, -0.0219]], grad_fn=<AddmmBackward>) >>> out.shape
        torch.Size([5, 2])
        """
        for i, layer in enumerate(self.layers):
            x = (
                torch.nn.functional.gelu(layer(x))
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


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
            * (-math.log(float(2 * max_val)) / dims_add)
        )
        self.register_buffer("div_term", div_term)
        self.static_size = static_size

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
        >>> x2 = torch.zeros((5, 2, 20))
        >>> encoder = ConcatenationEncoder(10, 0.1, 10)
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


class CosineLoss(torch.nn.CosineSimilarity):
    """CosineLoss Implements a simple cosine similarity based loss."""

    def __init__(self, *args, **kwargs) -> None:
        """__init__ Instantiates the class.

        All arguments are passed to `torch.nn.CosineSimilarity`
        """
        super().__init__(*args, **kwargs)

    def forward(self, truth: Tensor, prediction: Tensor) -> Tensor:
        """Forward calculates the loss.

        [extended_summary]

        Parameters
        ----------
        truth : Tensor
        prediction : Tensor

        Returns
        -------
        Tensor

        Examples
        --------
        >>> loss = CosineLoss(dim=1, eps=1e-4)
        >>> loss(torch.ones([1,2,5]), torch.zeros([1,2,5]))
        tensor([[1., 1., 1., 1., 1.]])
        >>> loss(torch.ones([1,2,5]), 5*torch.zeros([1,2,5]))
        tensor([[1., 1., 1., 1., 1.]])
        >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
        tensor([[0., 0., 0., 0., 0.]])
        """
        out = super().forward(truth, prediction)
        out = 1 - out
        return out


class _PeptideTransformerEncoder(torch.nn.Module):
    def __init__(
        self, ninp: int, dropout: float, nhead: int, nhid: int, layers: int
    ) -> None:
        super().__init__()

        # Positional encoding section
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(
            ninp, dropout, max_len=constants.MAX_SEQUENCE * 2
        )

        # Aminoacid encoding layer
        self.aa_encoder = torch.nn.Embedding(constants.AAS_NUM + 1, ninp, padding_idx=0)
        # PTM encoder
        self.mod_encoder = torch.nn.Embedding(
            len(constants.MODIFICATION) + 1, ninp, padding_idx=0
        )

        # Transformer encoder sections
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, layers)

        # Weight Initialization
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        ptm_initrange = initrange * 0.01
        torch.nn.init.uniform_(self.aa_encoder.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.mod_encoder.weight, -ptm_initrange, ptm_initrange)

    def forward(self, src: Tensor, mods: Tensor, debug: bool = False) -> Tensor:
        trans_encoder_mask = ~src.bool()
        if debug:
            print(f"TE: Shape of mask {trans_encoder_mask.size()}")

        src = self.aa_encoder(src.permute(1, 0))
        mods = self.mod_encoder(mods.permute(1, 0))
        src = src + mods

        src = src * math.sqrt(self.ninp)
        if debug:
            print(f"TE: Shape after encoder {src.shape}")
        src = self.pos_encoder(src)
        if debug:
            print(f"TE: Shape after pos encoder {src.shape}")

        trans_encoder_output = self.transformer_encoder(
            src, src_key_padding_mask=trans_encoder_mask
        )
        if debug:
            print(f"TE: Shape after trans encoder {trans_encoder_output.shape}")

        return trans_encoder_output


class _PeptideTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        ninp: int,
        nhead: int,
        layers: int,
        dropout: float,
        charge_dims_pct: float = 0.05,
        nce_dims_pct: float = 0.05,
    ) -> None:
        super().__init__()
        print(f"Creating TransformerDecoder ninp={ninp} nhead={nhead} layers={layers}")
        charge_dims = math.ceil(ninp * charge_dims_pct)
        nce_dims = math.ceil(ninp * nce_dims_pct)
        n_embeds = ninp - (charge_dims + nce_dims)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ninp, nhead=nhead, dropout=dropout
        )
        self.trans_decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.peak_decoder = MLP(ninp, ninp, output_dim=1, num_layers=3)

        print(
            f"Creating embedding for spectra of length {constants.NUM_FRAG_EMBEDINGS}"
        )
        self.trans_decoder_embedding = torch.nn.Embedding(
            constants.NUM_FRAG_EMBEDINGS, n_embeds
        )
        self.charge_encoder = ConcatenationEncoder(
            dims_add=charge_dims, dropout=dropout, max_val=10.0
        )
        self.nce_encoder = ConcatenationEncoder(
            dims_add=nce_dims, dropout=dropout, max_val=100.0
        )

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(
            self.trans_decoder_embedding.weight, -initrange, initrange
        )

    def forward(
        self,
        src: Tensor,
        charge: Tensor,
        nce: Tensor,
        debug: bool = False,
    ) -> Tensor:
        trans_decoder_tgt = self.trans_decoder_embedding.weight.unsqueeze(1)
        trans_decoder_tgt = trans_decoder_tgt.repeat(1, charge.size(0), 1)
        trans_decoder_tgt = self.charge_encoder(trans_decoder_tgt, charge, debug=debug)
        trans_decoder_tgt = self.nce_encoder(trans_decoder_tgt, nce)
        if debug:
            print(f"TD: Shape of query embedding {trans_decoder_tgt.shape}")

        spectra_output = self.trans_decoder(memory=src, tgt=trans_decoder_tgt)
        if debug:
            print(f"TD: Shape of the output spectra {spectra_output.shape}")

        spectra_output = self.peak_decoder(spectra_output)
        if debug:
            print(f"TD: Shape of the MLP spectra {spectra_output.shape}")

        spectra_output = spectra_output.squeeze(-1).permute(1, 0)
        if debug:
            print(f"TD: Shape of the permuted spectra {spectra_output.shape}")

        if self.training:
            spectra_output = torch.nn.functional.leaky_relu(spectra_output)
        else:
            spectra_output = torch.nn.functional.relu(spectra_output)

        return spectra_output


class PepTransformerModel(pl.LightningModule):
    """PepTransformerModel Predicts retention times and HCD spectra from peptides."""

    accepted_schedulers = ["plateau", "cosine", "onecycle"]
    __version__ = elfragmentador.__version__

    def __init__(
        self,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhid: int = 2024,
        ninp: int = 516,
        nhead: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-4,
        scheduler: str = "plateau",
        lr_ratio: Union[float, int] = 200,
        steps_per_epoch: None = None,
        *args,
        **kwargs,
    ) -> None:
        """__init__ Instantiates the class.

        Generates a new instance of the PepTransformerModel

        Parameters
        ----------
        num_decoder_layers : int, optional
            Number of layers in the transformer decoder, by default 6
        num_encoder_layers : int, optional
            Number of laters in the transformer encoder, by default 6
        nhid : int, optional
            Number of dimensions used in the feedforward networks inside
            the transformer encoder and decoders, by default 2024
        ninp : int, optional
            Number of features to pass to the transformer encoder.
            The embedding transforms the input to this input, by default 516
        nhead : int, optional
            Number of multi-attention heads in the transformer, by default 4
        dropout : float, optional
            dropout, by default 0.1
        lr : float, optional
            Learning rate, by default 1e-4
        scheduler : str, optional
            What scheduler to use, check the available ones with
            `PepTransformerModel.accepted_schedulers`, by default "plateau"
        lr_ratio : Union[float, int], optional
            For cosine annealing:
            Ratio of the initial learning rate to use with cosine annealing for
            instance a lr or 1 and a ratio of 10 would have a minimum learning
            rate of 0.1.

            For onecycle:
            Ratio of the initial lr and and maximum one,
            for instance if lr is 0.1 and ratio is 10, the max learn rate
            would be 1.0.

            by default 200
        steps_per_epoch : None, optional
            expected number of steps per epoch, used internally to calculate
            learning rates when using the oncecycle scheduler, by default None
        """
        super().__init__()
        self.save_hyperparameters()

        # Peptide encoder
        self.encoder = _PeptideTransformerEncoder(
            ninp=ninp,
            dropout=dropout,
            nhead=nhead,
            nhid=nhid,
            layers=num_encoder_layers,
        )

        # Peptide decoder
        self.decoder = _PeptideTransformerDecoder(
            ninp=ninp, nhead=nhead, layers=num_decoder_layers, dropout=dropout
        )

        # On this implementation, the rt predictor is a simple MLP
        # that combines the features from the transformer encoder
        self.rt_decoder = MLP(ninp, ninp, output_dim=1, num_layers=3)

        # Training related things
        self.mse_loss = torch.nn.MSELoss()
        self.angle_loss = CosineLoss(dim=1, eps=1e-4)
        self.lr = lr

        assert (
            scheduler in self.accepted_schedulers
        ), f"Passed scheduler '{scheduler} is not one of {self.accepted_schedulers}"
        self.scheduler = scheduler
        self.lr_ratio = lr_ratio
        self.steps_per_epoch = steps_per_epoch


    def forward(
        self,
        src: Tensor,
        nce: Tensor,
        mods: Optional[Tensor] = None,
        charge: Optional[Tensor] = None,
        debug: bool = False,
    ) -> PredictionResults:
        """Forward Generate predictions.

        Privides the function for the forward pass to the model.

        Parameters
        ----------
        src : Tensor
            Encoded pepide sequence [B, L] (view details)
        nce : Tensor
            float Tensor with the charges [B, 1]
        mods : Optional[Tensor], optional
            Encoded modification sequence [B, L], by default None
        charge : Optional[Tensor], optional
            long Tensor with the charges [B, 1], by default None
        debug : bool, optional
            When set, it will print (a lot) of the shapes of the intermediate
            tensors inside the model. By default False

        Details
        -------
            src:
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

        Returns
        -------
        PredictionResults
            A named tuple with two named results; irt and spectra
            iRT prediction [B, 1]
            Spectra prediction [B, self.num_queries]
        """
        if type(src) == dict and charge is None and mods is None:
            charge = src["charge"]
            mods = src.get("mods", None)
            src = src["src"]

        if debug:
            print(
                f"PT: Shape of inputs src={src.shape},"
                f" mods={mods.shape if mods is not None else None},"
                f" nce={nce.shape}"
                f" charge={charge.shape}"
            )

        trans_encoder_output = self.encoder.forward(src=src, mods=mods, debug=debug)
        rt_output = self.rt_decoder.forward(trans_encoder_output)
        if debug:
            print(f"PT: Shape after RT decoder {rt_output.shape}")

        rt_output = rt_output.mean(dim=0)
        if debug:
            print(f"PT: Shape of RT output {rt_output.shape}")

        spectra_output = self.decoder.forward(
            src=trans_encoder_output, charge=charge, nce=nce, debug=debug
        )

        if debug:
            print(
                f"PT: Final Outputs of shapes {rt_output.shape}, {spectra_output.shape}"
            )

        return PredictionResults(rt_output, spectra_output)

    def batch_forward(
        self, inputs: TrainBatch, debug: bool = False
    ) -> PredictionResults:
        """batch_forward Forward function that takes a `TrainBatch` as an input.

        This function is a wrapper around forward but takes a named tuple as an
        input instead of the positional/keword arguments.

        Parameters
        ----------
        inputs : TrainBatch
            Named tuple (check the documentation of that object for which names)
        debug : bool, optional
            When set, it will print (a lot) of the shapes of the intermediate
            tensors inside the model. By default False

        Returns
        -------
        PredictionResults
            A named tuple with two named results; irt and spectra
            
        """
        def unsqueeze_if_needed(x, dims):
            if len(x.shape) != dims:
                if debug:
                    print(f"PT: Unsqueezing tensor of shape {x.shape}")
                x = x.unsqueeze(1)
            else:
                if debug:
                    print(f"PT: Skipping Unsqueezing tensor of shape {x.shape}")
            return x

        out = self.forward(
            src=unsqueeze_if_needed(inputs.encoded_sequence, 2),
            mods=unsqueeze_if_needed(inputs.encoded_mods, 2),
            nce=unsqueeze_if_needed(inputs.nce, 2),
            charge=unsqueeze_if_needed(inputs.charge, 2),
            debug=debug,
        )
        return out

    def predict_from_seq(
        self, seq: str, charge: int, nce: float, debug: bool = False
    ) -> PredictionResults:
        """predict_from_seq Predicts spectra from a sequence as a string.

        Utility method that gets a sequence as a string, encodes it internally
        to the correct input form and outputs the predicted spectra.

        Note that the spectra is not decoded as an output, please check
        `elfragmentador.encoding_decoding.decode_fragment_tensor` for the
        decoding.

        The irt is scaled by 100 and is in the Biognosys scale.

        TODO: consider if the output should be decoded ...

        Parameters
        ----------
        seq : str
            Sequence to use for prediction, supports modifications in the form
            of S[PHOSPHO], S[+80] and T[181]
        charge : int
            Precursor charge to be assumed during the fragmentation
        nce : float
            Normalized collision energy to use during the prediction
        debug : bool, optional
            When set, it will print (a lot) of the shapes of the intermediate
            tensors inside the model. By default False

        Returns
        -------
        PredictionResults
            A named tuple with two named results; irt and spectra

        Examples
        --------
        >>> pl.seed_everything(42)
        42
        >>> my_model = PepTransformerModel() # Or load the model from a checkpoint
        Creating TransformerDecoder ninp=516 nhead=4 layers=6
        Creating embedding for spectra of length 174
        >>> _ = my_model.eval()
        >>> my_model.predict_from_seq("MYPEPT[PHOSPHO]IDEK", 3, 27)
        PredictionResults(irt=tensor([-0.1290], grad_fn=<SqueezeBackward1>), \
spectra=tensor([0.1503, ... 0.1528], grad_fn=<SqueezeBackward1>))
        >>> my_model.predict_from_seq("MYPEPT[PHOSPHO]IDEK", 3, 27, debug=True)
        >>PT: PEPTIDE INPUT Shape of peptide inputs torch.Size([1, 30]), torch.Size([1, 1])
        PT: Shape of inputs src=torch.Size([1, 30]), mods=torch.Size([1, 30]), nce=torch.Size([1, 1]) charge=torch.Size([1, 1])
        TE: Shape of mask torch.Size([1, 30])
        TE: Shape after encoder torch.Size([30, 1, 516])
        TE: Shape after pos encoder torch.Size([30, 1, 516])
        TE: Shape after trans encoder torch.Size([30, 1, 516])
        PT: Shape after RT decoder torch.Size([30, 1, 1])
        PT: Shape of RT output torch.Size([1, 1])
        CE: Shape of inputs val=torch.Size([1, 1]) x=torch.Size([174, 1, 464])
        CE: Shape before concat e=torch.Size([174, 1, 26]) x=torch.Size([174, 1, 464])
        CE: Shape after concat x=torch.Size([174, 1, 490])
        TD: Shape of query embedding torch.Size([174, 1, 516])
        TD: Shape of the output spectra torch.Size([174, 1, 516])
        TD: Shape of the MLP spectra torch.Size([174, 1, 1])
        TD: Shape of the permuted spectra torch.Size([1, 174])
        PT: Final Outputs of shapes torch.Size([1, 1]), torch.Size([1, 174])
        PredictionResults(irt=tensor([-0.1290], grad_fn=<SqueezeBackward1>), \
spectra=tensor([0.1503, ... 0.1528], grad_fn=<SqueezeBackward1>))
        """
        encoded_seq, encoded_mods = encoding_decoding.encode_mod_seq(seq)

        src = torch.Tensor(encoded_seq).unsqueeze(0).long()
        mods = torch.Tensor(encoded_mods).unsqueeze(0).long()
        in_charge = torch.Tensor([charge]).unsqueeze(0).long()
        in_nce = torch.Tensor([nce]).unsqueeze(0).float()

        if debug:
            print(
                f">>PT: PEPTIDE INPUT Shape of peptide"
                f" inputs {src.shape}, {in_charge.shape}"
            )

        out = self.forward(
            src=src, charge=in_charge, mods=mods, nce=in_nce, debug=debug
        )
        out = PredictionResults(*[x.squeeze(0) for x in out])

        return out

    @staticmethod
    def add_model_specific_args(parser: _ArgumentGroup) -> _ArgumentGroup:
        """add_model_specific_args Adds arguments to a parser.

        It is used to add the command line arguments for the training/generation
        of the model.

        Parameters
        ----------
        parser : _ArgumentGroup
            An argparser parser (anything that has the `.add_argument` method)
            to which the arguments will be added

        Returns
        -------
        _ArgumentGroup
            Same parser with the added arguments
        """
        parser.add_argument(
            "--num_queries",
            default=150,
            type=int,
            help="Expected encoding length of the spectra",
        )
        parser.add_argument(
            "--num_decoder_layers",
            default=6,
            type=int,
            help="Number of sub-encoder-layers in the encoder",
        )
        parser.add_argument(
            "--num_encoder_layers",
            default=6,
            type=int,
            help="Number of sub-encoder-layers in the decoder",
        )
        parser.add_argument(
            "--nhid",
            default=1024,
            type=int,
            help="Dimension of the feedforward networks",
        )
        parser.add_argument(
            "--ninp",
            default=516,
            type=int,
            help="Number of input features to the transformer encoder",
        )
        parser.add_argument(
            "--nhead", default=12, type=int, help="Number of attention heads"
        )
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument(
            "--scheduler",
            default="plateau",
            type=str,
            help=(
                "Scheduler to use during training, "
                f"either of {PepTransformerModel.accepted_schedulers}"
            ),
        )
        parser.add_argument(
            "--lr_ratio",
            default=200.0,
            type=float,
            help=(
                "For cosine annealing: "
                "Ratio of the initial learning rate to use with cosine annealing"
                " for instance a lr or 1 and a ratio of 10 would have a minimum"
                " learning rate of 0.1\n"
                "For onecycle: "
                "Ratio of the initial lr and and maximum one, "
                "for instance if lr is 0.1 and ratio is 10, the max learn rate"
                "would be 1.0"
            ),
        )
        return parser

    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[List[AdamW], List[Dict[str, Union[ReduceLROnPlateau, str]]]],
        Tuple[List[AdamW], List[Dict[str, Union[CosineAnnealingWarmRestarts, str]]]],
        Tuple[List[AdamW], List[Dict[str, Union[OneCycleLR, str]]]],
    ]:
        """configure_optimizers COnfigures the optimizers for training.

        It is internally used by pytorch_lightning during training, so far I
        implemented 3 options (set when making the module).

        OneCycleLR seems to give the best results overall in the least amount
        of time. The only tradeoff that I see is that resuming training does
        not seem to be really easy.

        Check the pytorch_lightning documentation to see how this is used in the
        training loop

        Returns
        -------
        Union[
            Tuple[List[AdamW], List[Dict[str, Union[ReduceLROnPlateau, str]]]],
            Tuple[List[AdamW], List[Dict[str, Union[CosineAnnealingWarmRestarts, str]]]],
            Tuple[List[AdamW], List[Dict[str, Union[OneCycleLR, str]]]],
        ]
            Two lists, one containing the optimizer and another contining the
            scheduler.

        Raises
        ------
        ValueError
            Raised when a scheduler that is not supported is requested.
            If you want to use another one, please over-write this method
            or make a subclass with the modification. (PRs are also welcome)

        """
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.scheduler == "plateau":
            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=2, verbose=True
                ),
                "interval": "epoch",
                "monitor": "v_l",
            }
        elif self.scheduler == "cosine":
            assert self.lr_ratio > 1
            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt,
                    T_0=1,
                    T_mult=2,
                    eta_min=self.lr / self.lr_ratio,
                    last_epoch=-1,
                    verbose=False,
                ),
                "interval": "step",
            }
        elif self.scheduler == "onecycle":
            assert self.steps_per_epoch is not None, "Please set steps_per_epoch"
            if self.trainer.max_epochs == 1000:
                warnings.warn("Max epochs was 1000, make sure you want this")
            if self.lr_ratio > 20:
                warnings.warn(
                    f"Provided LR ratio '{self.lr_ratio}' seems a lil high,"
                    " make sure you want that for the OneCycleLR scheduler"
                )
                time.sleep(3)  # just so the user has time to see the message...
            max_lr = self.lr * self.lr_ratio
            print(
                f">> Scheduler setup: max_lr {max_lr}, "
                f"Max Epochs: {self.trainer.max_epochs}, "
                f"Steps per epoch: {self.steps_per_epoch}, "
                f"Accumulate Batches {self.trainer.accumulate_grad_batches}"
            )
            spe = self.steps_per_epoch // self.trainer.accumulate_grad_batches
            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=spe,
                ),
                "interval": "step",
            }

        else:
            raise ValueError(
                "Scheduler should be one of 'plateau' or 'cosine', passed: ",
                self.scheduler,
            )

        print(f"\n\n>>> Setting up schedulers:\n\n{scheduler_dict}")

        return [opt], [scheduler_dict]

    def _step(self, batch: TrainBatch, batch_idx: int) -> Dict[str, Tensor]:
        """Run main functionality during training an testing steps.

        Internally used in training and evaluation steps during the training
        loop in pytorch_lightning.

        Does inference, loss calculation, handling of missing values ...
        """
        yhat_irt, yhat_spectra = self.batch_forward(batch)
        yhat_irt = yhat_irt[~batch.norm_irt.isnan()]
        norm_irt = batch.norm_irt[~batch.norm_irt.isnan()]

        loss_irt = self.mse_loss(yhat_irt, norm_irt.float())
        loss_spectra = self.angle_loss(yhat_spectra, batch.encoded_spectra).mean()

        if len(norm_irt.data) == 0:
            total_loss = loss_spectra
        else:
            total_loss = (loss_irt + loss_spectra * 9) / 10

        out = {
            "l": total_loss,
            "irt_l": loss_irt,
            "spec_l": loss_spectra,
        }

        assert not torch.isnan(total_loss), print(
            f"Fail at... \n Loss: {total_loss},\n"
            f"\n loss_irt: {loss_irt}\n"
            f"\n loss_spectra: {loss_spectra}\n"
            f"\n yhat_spec: {yhat_spectra},\n"
            f"\n y_spec: {batch.encoded_spectra}\n"
            f"\n y_irt: {norm_irt}, {len(norm_irt.data)}"
        )

        return out

    def training_step(
        self, batch: TrainBatch, batch_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """See pytorch_lightning documentation."""
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"t_" + k: v for k, v in step_out.items()}
        log_dict.update({"LR": self.trainer.optimizers[0].param_groups[0]["lr"]})

        self.log_dict(
            log_dict,
            prog_bar=True,
        )

        return {"loss": step_out["l"]}

    def validation_step(
        self, batch: TrainBatch, batch_idx: Optional[int] = None
    ) -> None:
        """See pytorch_lightning documentation."""
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"v_" + k: v for k, v in step_out.items()}

        self.log_dict(
            log_dict,
            prog_bar=True,
        )
    
    __doc__ += "\n"*2 + __init__.__doc__ + "\n"*2 + forward.__doc__ 
