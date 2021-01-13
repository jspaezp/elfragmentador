import math

import torch
from torch import nn
import pytorch_lightning as pl

from argparse import ArgumentParser


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)
    From: https://github.com/facebookresearch/detr/blob/models/detr.py#L289
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (
                torch.nn.functional.relu(layer(x))
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


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
        >>> print(posencoder(torch.ones((2,1,20)).float())[..., 0:3])
        tensor([[[1.1111, 2.2222, 1.1111]],
                [[2.0461, 1.7114, 1.5419]]])

        >>> print(posencoder(torch.ones((1,2,20)).float())[..., 0:3])
        tensor([[[1.1111, 2.2222, 1.1111],
                 [1.1111, 2.2222, 1.1111]]])

    Therfore encoding are (seq_length, batch, encodings)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PepTransformerModel(pl.LightningModule):
    def __init__(
        self,
        max_charge=6,
        num_queries=150,
        num_decoder_layers=6,
        num_encoder_layers=6,
        ntoken=26,
        nhid=1024,
        max_len=50,
        ninp=516,
        nhead=8,
        dropout=0.2,
        lr=1e-4,
        *args,
        **kwargs,
    ):
        """
        Parameters:
            num_queries:
                number of outputs to generate, should be the number of indices in the
                prediction matrix for the ions.
        """

        super().__init__()
        self.save_hyperparameters()

        # Positional encoding section
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=max_len)

        # Aminoacid encoding layer
        self.encoder = torch.nn.Embedding(ntoken + 1, ninp, padding_idx=0)

        # Transformer encoder sections
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )

        # On this implementation, the rt predictor is simply a set of linear layers
        # that combines the features from the transformer encoder
        # TODO check if this 2 layer approach is actually better than a single linear layer
        # or wethern an MLP would be better
        self.rt_decoder = MLP(ninp, ninp, output_dim=1, num_layers=3)
        """
        self.rt_decoder = torch.nn.Sequential(
            *[torch.nn.Linear(ninp, ninp // 2), torch.nn.Linear(ninp // 2, 1)]
        )
        """

        # Transformer decoder section
        decoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=nhead)
        transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        self.trans_decoder = transformer_decoder
        self.peak_decoder = MLP(ninp, ninp, output_dim=1, num_layers=3)

        # Input to the decoder layer (input and output is the same size in tranformers)
        # TODO change name because it does not really encode stuff
        self.trans_decoder_embedding = torch.nn.Embedding(num_queries, ninp)
        self.max_charge = max_charge

        # Weight Initialization
        self.init_weights()

        # Training related things
        self.loss = torch.nn.MSELoss()
        self.lr = lr

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--max_charge",
            default=6,
            type=int,
            help="Maximum precursor charge to be expected",
        )
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
            "--ntoken", default=26, type=int, help="number of distinc aminoacid entries"
        )
        parser.add_argument(
            "--nhid",
            default=1024,
            type=int,
            help="Dimension of the feedforward networks",
        )
        parser.add_argument(
            "--max_len",
            default=50,
            type=int,
            help="Maximum number of positions in the positional encoder",
        )
        parser.add_argument(
            "--ninp",
            default=516,
            type=int,
            help="number of input/output features in the transformer layers",
        )
        parser.add_argument(
            "--nhead", default=12, type=int, help="Number of attention heads"
        )
        parser.add_argument("--dropout", default=0.1, type=int)
        parser.add_argument("--lr", default=1e-4, type=int)
        return parser

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, verbose=True
        )

        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=self.lr/50,
            last_epoch=-1,
            verbose=False)

        """

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, src, charge, debug=False):
        """
        Parameters:
            src: Encoded pepide sequence [B, L] (view details)
            charge: Tensor with the charges [B, 1]

        Details:
            src:
                The peptide is enconded as integers for the aminoacid.
                "AAA" enconded for a max length of 5 would be
                torch.Tensor([ 1,  1,  1,  0,  0]).long()
            charge:
                A tensor corresponding to the charges of each of the
                peptide precursors (long)

        Returns:
            iRT prediction [B, 1]
            Spectra prediction [B, self.num_queries]

        """
        trans_encoder_mask = ~src.bool()
        if debug:
            print(f"Shape of mask {trans_encoder_mask.size()}")

        src = src.permute(1, 0)
        src = self.encoder(src) * math.sqrt(self.ninp)
        if debug:
            print(f"Shape after encoder {src.shape}")
        src = self.pos_encoder(src)
        if debug:
            print(f"Shape after pos encoder {src.shape}")

        trans_encoder_output = self.transformer_encoder(
            src, src_key_padding_mask=trans_encoder_mask
        )
        if debug:
            print(f"Shape after trans encoder {trans_encoder_output.shape}")

        rt_output = self.rt_decoder(trans_encoder_output)
        if debug:
            print(f"Shape after decoder {rt_output.shape}")

        rt_output = rt_output.mean(dim=0)
        if debug:
            print(f"Shape of RT output {rt_output.shape}")

        trans_decoder_tgt = self.trans_decoder_embedding.weight.unsqueeze(1)
        trans_decoder_tgt = trans_decoder_tgt * (charge.unsqueeze(0) / self.max_charge)
        if debug:
            print(f"Shape of query embedding {trans_decoder_tgt.shape}")

        # tgt has to be of shape (num_ions, batch, embedding)
        spectra_output = self.trans_decoder(
            memory=trans_encoder_output,
            tgt=trans_decoder_tgt,
        )
        if debug:
            print(f"Shape of the output spectra {spectra_output.shape}")

        spectra_output = self.peak_decoder(spectra_output)
        # spectra_output = spectra_output.mean(axis=2).permute(1, 0)
        if debug:
            print(f"Shape of the MLP spectra {spectra_output.shape}")

        spectra_output = spectra_output.squeeze().permute(1, 0)
        if debug:
            print(f"Shape of the permuted spectra {spectra_output.shape}")

        return rt_output, spectra_output

    def training_step(self, batch, batch_idx=None):
        encoded_sequence, charge, encoded_spectra, norm_irt = batch
        yhat_irt, yhat_spectra = self(encoded_sequence, charge)

        assert not all(torch.isnan(yhat_irt)), print(yhat_irt.mean())
        assert not all(torch.isnan(yhat_spectra).flatten()), print(yhat_spectra.mean())

        loss_irt = self.loss(yhat_irt.float(), norm_irt.float())
        loss_spectra = self.loss(yhat_spectra.float(), encoded_spectra.float())

        total_loss = loss_irt + (10 * loss_spectra)

        self.log_dict(
            {
                "train_loss": total_loss,
                "train_irt_loss": loss_irt,
                "train_spec_loss": loss_spectra,
            },
            prog_bar=True,
        )

        assert not torch.isnan(total_loss), print(
            f"Fail at Loss: {total_loss},\n"
            f" yhat: {total_loss},\n"
            f" y_spec: {encoded_spectra}\n"
            f" y_irt: {norm_irt}"
        )

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx=None):
        encoded_sequence, charge, encoded_spectra, norm_irt = batch
        yhat_irt, yhat_spectra = self(encoded_sequence, charge)

        loss_irt = self.loss(yhat_irt, norm_irt)
        loss_spectra = self.loss(yhat_spectra, encoded_spectra)
        total_loss = loss_irt + (10 * loss_spectra)

        self.log_dict(
            {
                "val_loss": total_loss,
                "val_irt_loss": loss_irt,
                "val_spec_loss": loss_spectra,
            },
            prog_bar=True,
        )


class TransformerModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        ntoken=26,
        nout=1,
        ninp=512,
        nhead=8,
        nhid=1024,
        nlayers=4,
        dropout=0.2,
        max_len=50,
    ):
        """
        Args:
            ntoken: the number of tokens in the dict
            nout: number of output features
            ninp: the number of expected features in the input of the transformer encoder.
            nhead: the number of heads in the multiheadattention models.
            nhid: the dimension of the feedforward network model (1024).
            num_layers: the number of sub-encoder-layers in the encoder
            dropout: the dropout value (default=0.1).
            nout: the number of out predictions
            max_len: maximum length of the sequences


        """
        super().__init__()
        self.model_type = "Transformer"

        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=max_len)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken + 1, ninp, padding_idx=0)
        self.ninp = ninp
        self.decoder = torch.nn.Linear(ninp, nout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # torch.nn.init.zeros_(self.decoder.weight)
        # torch.nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, debug=False):
        mask = ~src.bool()
        if debug:
            print(f"Shape of mask {mask.size()}")

        src = src.permute(1, 0)
        src = self.encoder(src) * math.sqrt(self.ninp)
        if debug:
            print(f"Shape after encoder {src.shape}")
        src = self.pos_encoder(src)
        if debug:
            print(f"Shape after pos encoder {src.shape}")

        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        if debug:
            print(f"Shape after trans encoder {output.shape}")

        output = self.decoder(output)
        if debug:
            print(f"Shape after decoder {output.shape}")

        output = output.mean(dim=0)

        return output


class LitTransformer(pl.LightningModule):
    def __init__(self, lr=0.0001, *args, **kwargs):
        super().__init__()
        self.transformer = TransformerModel(ntoken=26, nout=1, *args, **kwargs)
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, verbose=True
        )
        """
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=self.lr/50,
            last_epoch=-1,
            verbose=False)

        """

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x, debug=False):
        return self.transformer(x, debug=debug)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        yhat = self(x)

        assert not all(torch.isnan(yhat)), print(yhat.mean())

        loss = self.loss(yhat, y)

        self.log_dict({"train_loss": loss}, prog_bar=True)

        assert not torch.isnan(loss), print(
            f"Fail at Loss: {loss},\n yhat: {yhat},\n y: {y}"
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        yhat = self(x)

        loss = self.loss(yhat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
