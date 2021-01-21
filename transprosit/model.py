try:
    from typing import Optional, Union, Literal

    LiteralFalse = Literal[False]
except ImportError:
    # Python pre-3.8 compatibility
    from typing import Union, NewType

    LiteralFalse = NewType("LiteralFalse", bool)

import warnings
import math

import torch
from torch import Tensor, nn
import pytorch_lightning as pl

from argparse import ArgumentParser

import transprosit
from transprosit import constants
from transprosit import encoding_decoding


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)
    From: https://github.com/facebookresearch/detr/blob/models/detr.py#L289
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = (
                torch.nn.functional.gelu(layer(x))
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


class ConcatenationEncoder(torch.nn.Module):
    """
    ConcatenationEncoder concatenates information into the embedding

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
        max_val: float = 200.0,
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
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the encoder model (required).
            val: value to be encoded into the sequence (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            val: [batch size, 1]
            output: [sequence length, batch size, embed_dim + added_dims]
        Examples:
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
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
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
        static_size: bool = False,
    ) -> None:
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
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> x = torch.ones((1,2,20)).float()
            >>> pos_encoder = PositionalEncoding(20, 0.1, max_len=20)
            >>> output = pos_encoder(x)
        """

        if self.static_size:
            end_position = self.static_size
        else:
            end_position = x.size(0)

        x = x + self.pe[:end_position, :]
        return self.dropout(x)


class CosineLoss(torch.nn.CosineSimilarity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, truth, prediction):
        out = super().forward(truth, prediction)
        out = 1 - out
        return out


class PeptideTransformerEncoder(torch.nn.Module):
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
        ptm_initrange = initrange * 0.1
        torch.nn.init.uniform_(self.aa_encoder.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.mod_encoder.weight, -ptm_initrange, ptm_initrange)

    def forward(self, src: Tensor, mods: None = None, debug: bool = False) -> Tensor:
        trans_encoder_mask = ~src.bool()
        if debug:
            print(f"TE: Shape of mask {trans_encoder_mask.size()}")

        src = self.aa_encoder(src.permute(1, 0))
        if mods is not None:
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


class PeptideTransformerDecoder(torch.nn.Module):
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
        n_embeds = ninp - (charge_dims)

        warnings.warn("NCE has not been implemented yet ...  sorry")
        # ninp = ninp - (charge_dims + nce_dims)
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
        self, src: Tensor, charge: Tensor, nce: None = None, debug: bool = False
    ) -> Tensor:
        trans_decoder_tgt = self.trans_decoder_embedding.weight.unsqueeze(1)
        trans_decoder_tgt = trans_decoder_tgt.repeat(1, charge.size(0), 1)
        trans_decoder_tgt = self.charge_encoder(trans_decoder_tgt, charge, debug=debug)
        if nce is not None:
            raise NotImplementedError(
                "Sorry, I have not implemented NCE"
                " (mainly due to the dataset, all the code is ready for it ...)"
            )
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

        return torch.nn.functional.leaky_relu(spectra_output)


class PepTransformerModel(pl.LightningModule):
    accepted_schedulers = ["plateau", "cosine"]
    __version__ = transprosit.__version__

    def __init__(
        self,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhid: int = 1024,
        ninp: int = 516,
        nhead: int = 8,
        dropout: float = 0.2,
        lr: float = 1e-4,
        scheduler: str = "plateau",
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters:
            num_queries:
                number of outputs to generate, should be the number of indices in the
                prediction matrix for the ions.
        """

        super().__init__()
        self.save_hyperparameters()

        # Peptide encoder
        self.encoder = PeptideTransformerEncoder(
            ninp=ninp,
            dropout=dropout,
            nhead=nhead,
            nhid=nhid,
            layers=num_encoder_layers,
        )

        # Peptide decoder
        self.decoder = PeptideTransformerDecoder(
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

    def forward(self, src: torch.long, charge=None, mods=None, debug=False):
        """
        Parameters:
            src: Encoded pepide sequence [B, L] (view details)
            charge: Tensor with the charges [B, 1]

        Details:
            src:
                The peptide is encoded as integers for the aminoacid.
                "AAA" encoded for a max length of 5 would be
                torch.Tensor([ 1,  1,  1,  0,  0]).long()
            charge:
                A tensor corresponding to the charges of each of the
                peptide precursors (long)
            mods:
                Modifications encoded as integers

        Returns:
            iRT prediction [B, 1]
            Spectra prediction [B, self.num_queries]

        """

        if type(src) == dict and charge is None and mods is None:
            charge = src["charge"]
            mods = src.get("mods", None)
            src = src["src"]

        trans_encoder_output = self.encoder(src, mods=mods, debug=debug)
        rt_output = self.rt_decoder(trans_encoder_output)
        if debug:
            print(f"PT: Shape after RT decoder {rt_output.shape}")

        rt_output = rt_output.mean(dim=0)
        if debug:
            print(f"PT: Shape of RT output {rt_output.shape}")

        spectra_output = self.decoder(trans_encoder_output, charge, debug=debug)

        return rt_output, spectra_output

    def predict_from_seq(self, seq: str, charge: int, mods=None, debug: bool = False):
        src = torch.Tensor(encoding_decoding.encode_mod_seq(seq)).unsqueeze(0).long()
        in_charge = torch.Tensor([charge]).unsqueeze(0).long()

        if debug:
            print(
                f">>PT: PEPTIDE INPUT Shape of peptide inputs {src.shape}, {in_charge.shape}"
            )

        if mods is not None:
            raise NotImplementedError(
                "Sorry, have not implemented PTMS on this input ... yet"
            )

        out = self(src=src, charge=in_charge, mods=mods, debug=debug)
        out = tuple([x.squeeze(0) for x in out])

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
        return parser

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=2, verbose=True
            )
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=1, T_mult=2, eta_min=self.lr / 50, last_epoch=-1, verbose=False
            )
        else:
            raise ValueError(
                "Scheduler should be one of 'plateau' or 'cosine', passed: ",
                self.scheduler,
            )

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "v_l"}

    def _step(self, batch, batch_idx):
        encoded_sequence, charge, encoded_spectra, norm_irt = batch
        yhat_irt, yhat_spectra = self(encoded_sequence, charge)
        yhat_irt = yhat_irt[~norm_irt.isnan()]
        norm_irt = norm_irt[~norm_irt.isnan()]

        loss_irt = self.mse_loss(yhat_irt, norm_irt.float())
        loss_spectra = self.angle_loss(yhat_spectra, encoded_spectra).mean()

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
            f"\n y_spec: {encoded_spectra}\n"
            f"\n y_irt: {norm_irt}, {len(norm_irt.data)}"
        )

        return out

    def training_step(self, batch, batch_idx=None):
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"t_" + k: v for k, v in step_out.items()}

        self.log_dict(
            log_dict,
            prog_bar=True,
        )

        return {"loss": step_out["l"]}

    def validation_step(self, batch, batch_idx=None):
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"v_" + k: v for k, v in step_out.items()}

        self.log_dict(
            log_dict,
            prog_bar=True,
        )
