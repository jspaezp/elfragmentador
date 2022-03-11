import copy
import logging

try:
    from typing import Dict, List, Literal, Optional, Tuple, Union

    LiteralFalse = Literal[False]
except ImportError:
    # Python pre-3.8 compatibility
    from typing import Dict, List, NewType, Optional, Tuple, Union

    LiteralFalse = NewType("LiteralFalse", bool)

import math
import time
import warnings
from argparse import _ArgumentGroup

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import uniplot
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)

import elfragmentador
from elfragmentador import constants
from elfragmentador.math_utils import MissingDataAverager, polyfit
from elfragmentador.metrics import CosineLoss, MetricCalculator, SpectralAngleLoss
from elfragmentador.named_batches import ForwardBatch, PredictionResults, TrainBatch
from elfragmentador.nn_encoding import AASequenceEmbedding, ConcatenationEncoder
from elfragmentador.spectra import Spectrum
from elfragmentador.utils import torch_batch_from_seq


# TODO refactor this massive models into manageable sections
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


class _IRTDecoder(nn.Module):
    def __init__(self, d_model, dim_feedforward=224, nhead=4, n_layers=3, dropout=0.05):
        super().__init__()
        """Decode iRTs

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

        self.aa_embed = AASequenceEmbedding(d_model=d_model)
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


class _PeptideTransformerEncoder(torch.nn.Module):
    def __init__(
        self, d_model: int, dropout: float, nhead: int, nhid: int, layers: int
    ) -> None:
        super().__init__()

        # Aminoacid embedding
        self.aa_encoder = AASequenceEmbedding(d_model=d_model)

        # Transformer encoder sections
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)

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

        x = self.aa_encoder(seq=seq, mods=mods)
        # x shape [S, N, d_model]

        trans_encoder_output = self.transformer_encoder(
            x, src_key_padding_mask=trans_encoder_mask
        )
        # trans_encoder_output shape [S, N, d_model]

        return trans_encoder_output, trans_encoder_mask


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
        super().__init__()
        logging.info(
            (
                f"Creating TransformerDecoder"
                f" nhid={nhid}, "
                f"d_model={d_model} "
                f"nhead={nhead} "
                f"layers={layers}"
            )
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

        logging.info(f"Creating embedding for spectra of length {num_outputs}")
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
        """
        Has to be implemente when subclassing.
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
        """
        Has to be implemente when subclassing.
        """
        trans_decoder_tgt = self.get_learnable_query(batch_size=memory.size(1))
        trans_decoder_tgt = self.preprocess_query(trans_decoder_tgt)

        output = self.decoder_forward(
            trans_decoder_tgt=trans_decoder_tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output


class _PeptideTransformerDecoder(_LearnableEmbedTransformerDecoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        nhid: int,
        layers: int,
        dropout: float,
        charge_dims_pct: float = 0.05,
        nce_dims_pct: float = 0.05,
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
            num_outputs=constants.NUM_FRAG_EMBEDINGS,
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


_model_sections = [
    "TransEncoder",
    "TransDecoder",
    "AAEmbedding",
    "MODEmbedding",
    "FragmentEmbedding",
    "FragmentFFN",
    "IrtDecoder",
]


class PepTransformerModel(pl.LightningModule):
    """
    PepTransformerModel Predicts retention times and HCD spectra from peptides.
    """

    accepted_schedulers = ["plateau", "cosine", "onecycle"]
    model_sections = _model_sections
    __version__ = elfragmentador.__version__

    def __init__(
        self,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhid: int = 2024,
        d_model: int = 516,
        nhead: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-4,
        scheduler: str = "plateau",
        lr_ratio: Union[float, int] = 200,
        steps_per_epoch: None = None,
        loss_ratio: float = 5,
        trainable_sections: List[str] = _model_sections,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ Instantiates the class.

        Generates a new instance of the PepTransformerModel

        Parameters:
            num_decoder_layers : int, optional
                Number of layers in the transformer decoder, by default 6
            num_encoder_layers : int, optional
                Number of laters in the transformer encoder, by default 6
            nhid : int, optional
                Number of dimensions used in the feedforward networks inside
                the transformer encoder and decoders, by default 2024
            d_model : int, optional
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
            loss_ratio: float, optional
                The ratio of the spectrum to retention time loss to use when adding
                before passing to the optimizer. Higher values mean more weight to
                spectra with respect to the retention time. By default 5
        """
        super().__init__()
        self.save_hyperparameters()
        logging.info(
            f"num_decoder_layers {num_decoder_layers} "
            f"num_encoder_layers {num_encoder_layers} "
            f"nhid {nhid} d_model {d_model} "
            f"nhead {nhead} dropout {dropout}"
        )

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
        )

        # On this implementation, the rt predictor is a simple MLP
        # that combines the features from the transformer encoder

        self.irt_decoder = _IRTDecoder(
            d_model=d_model,
            dim_feedforward=nhid,
            nhead=nhead,
            n_layers=num_encoder_layers,
            dropout=dropout,
        )

        self.metric_calculator = MetricCalculator()
        self.mse_loss = nn.MSELoss(reduction="none")
        self.cosine_loss = CosineLoss(dim=1, eps=1e-8)
        self.angle_loss = SpectralAngleLoss(dim=1, eps=1e-8)

        # Training related things
        self.lr = lr

        assert (
            scheduler in self.accepted_schedulers
        ), f"Passed scheduler '{scheduler} is not one of {self.accepted_schedulers}"
        self.scheduler = scheduler
        self.lr_ratio = lr_ratio
        self.steps_per_epoch = steps_per_epoch
        self.loss_ratio = loss_ratio

        self.model_sections = {
            "TransEncoder": self.encoder.transformer_encoder,
            "TransDecoder": self.decoder.trans_decoder,
            "AAEmbedding": self.encoder.aa_encoder.aa_encoder,
            "MODEmbedding": self.encoder.aa_encoder.mod_encoder,
            "FragmentEmbedding": self.decoder.trans_decoder_embedding,
            "FragmentFFN": self.decoder.peak_decoder,
            "IrtDecoder": self.irt_decoder,
        }

        # self.make_trainable_sections(trainable_sections)
        self.irt_metric = MissingDataAverager()
        self.loss_metric = MissingDataAverager()
        self.spectra_metric = MissingDataAverager()
        self.spectra_metric2 = MissingDataAverager()

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

    @staticmethod
    def torch_batch_from_seq(*args, **kwargs) -> ForwardBatch:
        torch_batch_from_seq.__doc__
        return torch_batch_from_seq(*args, **kwargs)

    def to_torchscript(self):
        """
        Convert the model to torchscript.
        """
        _fake_input_data_torchscript = self.torch_batch_from_seq(
            seq="MYM[OXIDATION]DIFIEDPEPTYDE", charge=3, nce=27.0
        )

        bkp_1 = self.decoder.nce_encoder.static_size
        self.decoder.nce_encoder.static_size = constants.NUM_FRAG_EMBEDINGS

        bkp_2 = self.decoder.charge_encoder.static_size
        self.decoder.charge_encoder.static_size = constants.NUM_FRAG_EMBEDINGS

        script = super().to_torchscript(
            example_inputs=_fake_input_data_torchscript, method="trace"
        )

        self.decoder.nce_encoder.static_size = bkp_1
        self.decoder.charge_encoder.static_size = bkp_2

        return script

    def predict_from_seq(
        self,
        seq: str,
        charge: int,
        nce: float,
        as_spectrum=False,
        enforce_length=True,
    ) -> Union[PredictionResults, Spectrum]:
        """
        predict_from_seq Predicts spectra from a sequence as a string.

        Utility method that gets a sequence as a string, encodes it internally
        to the correct input form and outputs the predicted spectra.

        Note that the spectra is not decoded as an output, please check
        `elfragmentador.encoding_decoding.decode_fragment_tensor` for the
        decoding.

        The irt is scaled by 100 and is in the Biognosys scale.

        TODO: consider if the output should be decoded ...

        Parameters:
            seq (str):
                Sequence to use for prediction, supports modifications in the form
                of S[PHOSPHO], S[+80] and T[181]
            charge (int):
                Precursor charge to be assumed during the fragmentation
            nce (float):
                Normalized collision energy to use during the prediction
            as_spectrum (bool, optional):
                Wether to return a Spectrum object instead of the raw tensor predictions
                (Default value = False)

        Returns:
          PredictionResults: A named tuple with two named results; irt and spectra
          Spectrum: A spectrum object with the predicted spectrum

        Examples:
            >>> pl.seed_everything(42)
            42
            >>> my_model = PepTransformerModel() # Or load the model from a checkpoint
            >>> _ = my_model.eval()
            >>> my_model.predict_from_seq("MYPEPT[PHOSPHO]IDEK", 3, 27)
            PredictionResults(irt=tensor(..., grad_fn=<SqueezeBackward1>), \
            spectra=tensor([...], grad_fn=<SqueezeBackward1>))
            >>> out = my_model.predict_from_seq("MYPEPT[PHOSPHO]IDEK", 3, 27, \
            as_spectrum=True)
            >>> type(out)
            <class 'elfragmentador.spectra.Spectrum'>
            >>> # my_model.predict_from_seq("MYPEPT[PHOSPHO]IDEK", 3, 27)
        """

        in_batch = self.torch_batch_from_seq(
            seq, nce, charge, enforce_length=enforce_length
        )

        in_batch_dict = {k: v.to(self.device) for k, v in in_batch._asdict().items()}

        out = self.forward(**in_batch_dict)
        out = PredictionResults(
            **{k: x.squeeze(0).cpu() for k, x in out._asdict().items()}
        )
        logging.debug(out)

        # rt should be in seconds for spectrast ...
        # irt should be non-dimensional
        if as_spectrum:

            out = Spectrum.from_tensors(
                sequence_tensor=in_batch.seq.squeeze().numpy(),
                fragment_tensor=out.spectra / out.spectra.max(),
                mod_tensor=in_batch.mods.squeeze().numpy(),
                charge=charge,
                nce=nce,
                rt=float(out.irt),
                irt=float(out.irt),  # * 100,
            )

        return out

    @staticmethod
    def add_model_specific_args(parser: _ArgumentGroup) -> _ArgumentGroup:
        """
        add_model_specific_args Adds arguments to a parser.

        It is used to add the command line arguments for the training/generation
        of the model.

        Args:
            parser (_ArgumentGroup):
                An argparser parser (anything that has the `.add_argument` method) to
                which the arguments will be added

        Returns:
            _ArgumentGroup, the same parser with the added arguments
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
            "--d_model",
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
        parser.add_argument(
            "--loss_ratio",
            default=5.0,
            type=float,
            help=(
                "Ratio between the retention time and the spectrum loss"
                " (higher values mean more weight to the spectra loss"
                " with respect to the retention time loss)"
            ),
        )
        parser.add_argument(
            "--trainable_secions",
            nargs="+",
            type=str,
            default=PepTransformerModel.model_sections,
            help=(
                f"Sections of the model to train, "
                f"can be any subset of {PepTransformerModel.model_sections}"
            ),
        )

        return parser

    def make_trainable_sections(self, sections: List[str] = _model_sections) -> None:
        """
        Makes sections of the model trainable.

        It freezes the whole model and makes the specified sections trainable

        Args:
            sections (List[str]):
                A list containing the model sections that should be set as trainable
        """

        def set_grad_section(model_section, trainable=True):
            """
            Freezes or unfreezes a model section.

            Args:
              model_section:
              trainable: (Default value = True)

            Returns:
            """
            for param in model_section.parameters():
                param.requires_grad = trainable

        logging.warning("Freezing the model")
        set_grad_section(self, trainable=False)

        for section in sections:
            logging.warning(f"Unfreezing {section}")
            try:
                set_grad_section(self.model_sections[section], trainable=True)
            except KeyError as e:
                logging.error(
                    (
                        f"{e} not found, please provide to the trainable"
                        " sections one of {_model_sections}"
                    )
                )

    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[List[AdamW], List[Dict[str, Union[ReduceLROnPlateau, str]]]],
        Tuple[List[AdamW], List[Dict[str, Union[CosineAnnealingWarmRestarts, str]]]],
        Tuple[List[AdamW], List[Dict[str, Union[OneCycleLR, str]]]],
    ]:
        """
        configure_optimizers COnfigures the optimizers for training.

        It is internally used by pytorch_lightning during training, so far I
        implemented 3 options (set when making the module).

        OneCycleLR seems to give the best results overall in the least amount
        of time. The only tradeoff that I see is that resuming training does
        not seem to be really easy.

        Check the pytorch_lightning documentation to see how this is used in the
        training loop

        Returns:
          Two lists, one containing the optimizer and another contining the scheduler.
        """
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            betas=(0.9, 0.98),
        )

        if self.scheduler == "plateau":
            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=2, verbose=True
                ),
                "interval": "epoch",
                "monitor": "val_l",
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
            spe = self.steps_per_epoch // self.trainer.accumulate_grad_batches
            # 4k warmup steps / total number of steps
            pct_start = 4000 / (spe * self.trainer.max_epochs)

            logging.info(
                f">> Scheduler setup: max_lr {max_lr}, "
                f"Max Epochs: {self.trainer.max_epochs}, "
                f"Steps per epoch: {self.steps_per_epoch}, "
                f"SPE (after accum grad batches) {spe}, "
                f"Percent Warmup {pct_start}, "
                f"Accumulate Batches {self.trainer.accumulate_grad_batches}, "
            )

            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=opt,
                    max_lr=max_lr,
                    epochs=self.trainer.max_epochs,
                    pct_start=pct_start,
                    steps_per_epoch=spe,
                ),
                "interval": "step",
            }

        else:
            raise ValueError(
                "Scheduler should be one of 'plateau' or 'cosine', passed: ",
                self.scheduler,
            )
        # TODO check if using different optimizers for different parts of the
        # model would work better
        logging.info(f"\n\n>>> Setting up schedulers:\n\n{scheduler_dict}")

        return [opt], [scheduler_dict]

    def _step(
        self, batch: TrainBatch, batch_idx: int, layernorm=False
    ) -> Dict[str, Tensor]:
        """
        Run main functionality during training an testing steps.

        Internally used in training and evaluation steps during the training
        loop in pytorch_lightning.

        Does inference, loss calculation, handling of missing values ...
        """

        if isinstance(batch, list):
            batch = TrainBatch(*batch)

        yhat_irt, yhat_spectra = self.forward(
            seq=batch.seq, mods=batch.mods, nce=batch.nce, charge=batch.charge
        )

        truth_spectra = F.pad(
            batch.spectra,
            (0, constants.NUM_FRAG_EMBEDINGS - batch.spectra.size(1)),
            mode="constant",
        )

        if layernorm:
            scaling = 1 - (truth_spectra + 1e-8).mean(axis=0)
            grandmean = scaling.mean()

            yhat_spectra = scaling * yhat_spectra / grandmean
            truth_spectra = scaling * truth_spectra / grandmean

        loss_irt = self.mse_loss(
            yhat_irt[~batch.irt.isnan()], batch.irt[~batch.irt.isnan()]
        ).squeeze()
        loss_angle = self.angle_loss(yhat_spectra, truth_spectra).squeeze()
        loss_cosine = self.cosine_loss(yhat_spectra, truth_spectra).squeeze()

        loss_irt = (
            loss_irt.squeeze() * batch.weight.squeeze()[~batch.irt.squeeze().isnan()]
        )
        loss_irt = loss_irt.mean() / batch.weight[~batch.irt.isnan()].mean()

        loss_angle = loss_angle * batch.weight.squeeze()
        loss_angle = loss_angle.mean() / batch.weight.mean()

        loss_cosine = loss_cosine * batch.weight.squeeze()
        loss_cosine = loss_cosine.mean() / batch.weight.mean()

        total_loss = loss_angle
        if not torch.any(torch.isnan(loss_irt)):
            # total_loss = loss_irt + (total_loss * self.loss_ratio)
            # total_loss = total_loss / (self.loss_ratio + 1)
            total_loss = loss_irt + total_loss
        else:
            logging.warning(
                (
                    f"Skipping addition of irt loss on batch {batch_idx} "
                    f"with value {loss_irt.flatten()}, "
                    f"preds: {yhat_irt.flatten()}"
                )
            )

        losses = {
            "l": total_loss,
            "irt_l": loss_irt,
            "spec_l": loss_cosine,
            "spec_l2": loss_angle,
        }

        if torch.isnan(total_loss):
            if hasattr(self, "num_failed"):
                self.num_failed += 1

            else:
                self.num_failed = 1

            logging.error(
                f"Fail {self.num_failed} at... batch {batch_idx} \n"
                f" Loss: {total_loss},\n"
                f"\n loss_irt: {loss_irt.flatten()}\n"
                f"\n loss_spectra: {loss_cosine.flatten()}\n"
                f"\n yhat_spec: {yhat_spectra.flatten()},\n"
                f"\n y_spec: {batch.spectra.flatten()}\n"
                f"\n y_irt: {batch.irt.flatten()}, {len(batch.irt.data)}"
                f"\n yhat_irt: {yhat_irt.flatten()}"
            )

            if self.num_failed > 2:
                torch.save(self.cpu().state_dict(), "broken_state.pt")
                logging.error(self.cpu().state_dict())

                logging.error("last succesfull state:")
                logging.error(self.last_state)
                raise RuntimeError(
                    "Too many nan in a row, dumping state to 'broken_state.pt'"
                )

        else:
            self.num_failed = 0
            if batch_idx % 50 == 0:
                logging.debug("Saved state dict")
                self.last_state = copy.deepcopy(self.state_dict())

        return losses

    def training_step(
        self, batch: TrainBatch, batch_idx: Optional[int] = None
    ) -> Tensor:
        """
        See pytorch_lightning documentation.
        """
        step_out = self._step(batch, batch_idx=batch_idx, layernorm=False)
        log_dict = {"train_" + k: v for k, v in step_out.items()}
        log_dict.update({"LR": self.trainer.optimizers[0].param_groups[0]["lr"]})

        self.log_dict(
            log_dict,
            prog_bar=True,
            # reduce_fx=nanmean,
        )

        return step_out["l"]

    def on_train_start(self) -> None:
        logging.info("Weights before the start of the training epoch:")
        self.last_state = copy.deepcopy(self.state_dict())
        logging.info(self.last_state)
        return super().on_train_start()

    def validation_step(
        self, batch: TrainBatch, batch_idx: Optional[int] = None
    ) -> Tensor:
        """
        See pytorch_lightning documentation.
        """
        step_out = self._step(batch, batch_idx=batch_idx)

        self.irt_metric.update(step_out["irt_l"])
        self.loss_metric.update(step_out["l"])
        self.spectra_metric.update(step_out["spec_l"])
        self.spectra_metric2.update(step_out["spec_l2"])

        return step_out["l"]

    def validation_epoch_end(self, outputs: List[Tensor]) -> List[Tensor]:
        """
        See pytorch lightning documentation.
        """
        log_dict = {
            "val_irt_l": self.irt_metric.compute(),
            "val_l": self.loss_metric.compute(),
            "val_spec_l": self.spectra_metric.compute(),
            "val_spec_l2": self.spectra_metric2.compute(),
        }

        self.log_dict(
            log_dict,
            prog_bar=True,
        )

        self.irt_metric.reset()
        self.loss_metric.reset()
        self.spectra_metric.reset()
        self.spectra_metric2.reset()

        return super().validation_epoch_end(outputs)

    def _evaluation_step(
        self,
        batch: TrainBatch,
        batch_idx: Optional[int],
    ) -> Dict[str, Tensor]:
        """
        Run main functionality during training an testing steps.

        Internally used in training and evaluation steps during the training
        loop in pytorch_lightning.

        Does inference, loss calculation, handling of missing values ...
        """
        if isinstance(batch, list):
            batch = TrainBatch(*batch)

        yhat_irt, yhat_spectra = self.forward(
            seq=batch.seq, mods=batch.mods, charge=batch.charge, nce=batch.nce
        )

        pred_out = PredictionResults(irt=yhat_irt, spectra=yhat_spectra)
        gt_out = PredictionResults(irt=batch.irt.float(), spectra=batch.spectra)

        loss_irt, loss_angle, loss_cosine = self.metric_calculator(pred_out, gt_out)

        losses = {
            "loss_irt": loss_irt,
            "loss_angle": loss_angle,
            "loss_cosine": loss_cosine,
        }

        return losses, pred_out

    def test_step(
        self, batch, batch_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Tensor], PredictionResults]:
        losses, pred_out = self._evaluation_step(batch=batch, batch_idx=batch_idx)
        return losses, pred_out.irt, batch.irt

    def test_epoch_end(self, results: List):
        self.metric_calculator.trainer = self.trainer
        self.metric_calculator.log_dict = self.log_dict
        return self.metric_calculator.test_epoch_end(results)

    def predict_step(self, batch: TrainBatch, batch_idx: Optional[int] = None):
        yhat_irt, yhat_spectra = self.forward(
            seq=batch.seq, mods=batch.mods, charge=batch.charge, nce=batch.nce
        )
        pred_out = PredictionResults(irt=yhat_irt, spectra=torch.relu(yhat_spectra))
        return pred_out

    def on_after_backward(self):

        msg = []
        global_step = self.global_step
        if (global_step % 50) == 0:
            for name, param in self.named_parameters():
                if "weight" in name and "norm" not in name:
                    if param.requires_grad:
                        try:
                            if param.grad is None:
                                raise AttributeError
                            if any(
                                x in name
                                for x in [
                                    "aa_encoder.weight",
                                    "mod_encoder.weight",
                                    "trans_decoder_embedding.weight",
                                ]
                            ):
                                val = param.grad.abs().mean()
                                if torch.any(torch.isnan(val)):
                                    logging.error(
                                        f"nan mean gradient for {name}: {param.grad}"
                                    )
                                self.log(name, val, prog_bar=True, on_step=True)
                        except AttributeError:
                            msg.append(name)
                        except ValueError:
                            msg.append(name)

        if len(msg) > 0:
            logging.warning(
                " ".join(msg) + "Did not have gradients in step {global_step}"
            )

    def on_train_epoch_end(self) -> None:
        evaluate_landmark_rt(self)
        return super().on_train_epoch_end()

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        mod = super().load_from_checkpoint(*args, **kwargs)
        evaluate_landmark_rt(mod)
        return mod


def evaluate_landmark_rt(model: PepTransformerModel):
    """
    evaluate_landmark_rt Checks the prediction of the model on the iRT
    peptides.

    Predicts all the procal and Biognosys iRT peptides and checks the correlation
    of the theoretical iRT values and the predicted ones

    Parameters
    ----------
    model : PepTransformerModel
        A model to test the predictions on
    """
    model.eval()
    real_rt = []
    pred_rt = []
    for seq, desc in constants.IRT_PEPTIDES.items():
        with torch.no_grad():
            out = model.predict_from_seq(seq, 2, 25, enforce_length=False)
            pred_rt.append(out.irt.numpy())
            real_rt.append(np.array(desc["irt"]))

    fit = polyfit(np.array(real_rt).flatten(), np.array(pred_rt).flatten())
    logging.info(fit)
    uniplot.plot(xs=np.array(real_rt).flatten(), ys=np.array(pred_rt).flatten())
    return fit
