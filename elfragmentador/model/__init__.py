from __future__ import annotations

import copy

try:
    from typing import Literal

    LiteralFalse = Literal[False]
except ImportError:
    # Python pre-3.8 compatibility
    from typing import NewType

    LiteralFalse = NewType("LiteralFalse", bool)

import time
import warnings
from argparse import _ArgumentGroup

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import uniplot
from loguru import logger
from ms2ml import Spectrum
from ms2ml.landmarks import IRT_PEPTIDES
from pytorch_lightning.utilities.model_summary import summarize
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)

import elfragmentador
from elfragmentador.config import CONFIG
from elfragmentador.math_utils import MissingDataAverager, polyfit
from elfragmentador.metrics import CosineLoss, MetricCalculator, SpectralAngleLoss
from elfragmentador.model.peptransformer import PepTransformerBase
from elfragmentador.named_batches import ForwardBatch, PredictionResults, TrainBatch
from elfragmentador.utils import torch_batch_from_seq


class PepTransformerModel(pl.LightningModule):
    """PepTransformerModel Predicts retention times and HCD spectra from peptides."""

    accepted_schedulers = ["plateau", "cosine", "onecycle"]
    __version__ = elfragmentador.__version__

    def __init__(
        self,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhid: int = 2024,
        d_model: int = 516,
        nhead: int = 4,
        dropout: float = 0.1,
        combine_embeds: bool = True,
        combine_encoders: bool = True,
        final_decoder: str = "linear",
        lr: float = 1e-4,
        scheduler: str = "plateau",
        lr_ratio: float | int = 200,
        steps_per_epoch: None = None,
        loss_ratio: float = 5,
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
            combine_embeds: bool, optional
                Whether the embeddings for modifications and sequences
                should be shared for irt and fragment predictions
            combine_encoders: bool = True,
                Whether the transformer encoders for for irt and
                fragments should be shared.
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
        self.ms2ml_config = CONFIG
        self.NUM_FRAGMENT_EMBEDDINGS = self.ms2ml_config.num_fragment_embeddings
        self.save_hyperparameters()
        logger.info(
            f"num_decoder_layers {num_decoder_layers} "
            f"num_encoder_layers {num_encoder_layers} "
            f"nhid {nhid} d_model {d_model} "
            f"nhead {nhead} dropout {dropout}"
        )
        self.main_model = PepTransformerBase(
            num_fragments=self.NUM_FRAGMENT_EMBEDDINGS,
            num_decoder_layers=num_decoder_layers,
            num_encoder_layers=num_encoder_layers,
            nhid=nhid,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            combine_embeds=combine_embeds,
            combine_encoders=combine_encoders,
            final_decoder=final_decoder,
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

        self.irt_metric = MissingDataAverager()
        self.loss_metric = MissingDataAverager()
        self.spectra_metric = MissingDataAverager()
        self.spectra_metric2 = MissingDataAverager()

    def summarize(self, max_depth=3):
        return summarize(self, max_depth)

    def forward(
        self,
        seq: Tensor,
        mods: Tensor,
        charge: Tensor,
        nce: Tensor,
    ):
        return self.main_model.forward(seq=seq, mods=mods, charge=charge, nce=nce)

    def predict_from_seq(
        self,
        seq: str,
        nce: float,
        as_spectrum=False,
    ) -> PredictionResults | Spectrum:
        return self.main_model.predict_from_seq(
            seq=seq,
            nce=nce,
            as_spectrum=as_spectrum,
        )

    @staticmethod
    def torch_batch_from_seq(*args, **kwargs) -> ForwardBatch:
        torch_batch_from_seq.__doc__
        return torch_batch_from_seq(*args, **kwargs)

    def to_torchscript(self):
        """
        Convert the model to torchscript.

        Example:
        >>> model = PepTransformerModel()
        >>> ts = model.to_torchscript()
        >>> type(ts)
        <class 'torch.jit._trace.TopLevelTracedModule'>
        """
        _fake_input_data_torchscript = self.torch_batch_from_seq(
            seq="MYM[U:35]DIFIEDPEPTYDE", charge=3, nce=27.0
        )

        backup_calculator = self.metric_calculator
        self.metric_calculator = None

        bkp_1 = self.main_model.decoder.nce_encoder.static_size
        self.main_model.decoder.nce_encoder.static_size = self.NUM_FRAGMENT_EMBEDDINGS
        bkp_2 = self.main_model.decoder.charge_encoder.static_size
        self.main_model.decoder.charge_encoder.static_size = (
            self.NUM_FRAGMENT_EMBEDDINGS
        )

        script = super().to_torchscript(
            example_inputs=_fake_input_data_torchscript, method="trace"
        )

        self.main_model.decoder.nce_encoder.static_size = bkp_1
        self.main_model.decoder.charge_encoder.static_size = bkp_2
        self.main_model.metric_calculator = backup_calculator

        return script

    @staticmethod
    def add_model_specific_args(parser: _ArgumentGroup) -> _ArgumentGroup:
        """
        Add_model_specific_args Adds arguments to a parser.

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
        parser.add_argument(
            "--combine_embeds",
            default=True,
            type=bool,
            help=(
                "Whether the embeddings for aminoacid and modifications"
                " should be shared between the irt and fragment sections"
            ),
        )
        parser.add_argument(
            "--combine_encoders",
            default=True,
            type=bool,
            help=(
                "Whether the encoders for aminoacid and modifications"
                " should be shared between the irt and fragment sections"
            ),
        )
        parser.add_argument(
            "--final_decoder",
            default="mlp",
            type=str,
            help=(
                "What kind of final layer should the docer have to"
                " output a single number, options are 'mlp' and 'linear'"
            ),
        )
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

        return parser

    @staticmethod
    def configure_scheduler_plateau(optimizer, lr_ratio):
        assert lr_ratio < 1
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=lr_ratio,
                patience=2,
                verbose=False,
            ),
            "interval": "epoch",
            "monitor": "val_l",
        }
        return scheduler_dict

    @staticmethod
    def configure_scheduler_cosine(optimizer, lr_ratio, min_lr):
        assert lr_ratio > 1
        scheduler_dict = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=1,
                T_mult=2,
                eta_min=min_lr,
                last_epoch=-1,
                verbose=False,
            ),
            "interval": "step",
        }
        return scheduler_dict

    @staticmethod
    def configure_scheduler_oncecycle(
        optimizer,
        lr_ratio,
        learning_rate,
        steps_per_epoch,
        accumulate_grad_batches,
        max_epochs,
    ):
        max_lr = learning_rate * lr_ratio
        spe = steps_per_epoch // accumulate_grad_batches
        pct_start = 0.3

        logger.info(
            f">> Scheduler setup: max_lr {max_lr}, "
            f"Max Epochs: {max_epochs}, "
            f"Steps per epoch: {steps_per_epoch}, "
            f"SPE (after accum grad batches) {spe}, "
            f"Percent Warmup {pct_start}, "
            f"Accumulate Batches {accumulate_grad_batches}, "
        )

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=max_lr,
                epochs=max_epochs,
                pct_start=pct_start,
                steps_per_epoch=spe,
            ),
            "interval": "step",
        }
        return scheduler_dict

    def configure_optimizers(
        self,
    ) -> (
        tuple[list[AdamW], list[dict[str, ReduceLROnPlateau | str]]]
        | tuple[list[AdamW], list[dict[str, CosineAnnealingWarmRestarts | str]]]
        | tuple[list[AdamW], list[dict[str, OneCycleLR | str]]]
    ):
        """
        Configure_optimizers COnfigures the optimizers for training.

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
            sched_dict = self.configure_scheduler_plateau(
                optimizer=opt, lr_ratio=self.lr_ratio
            )
        elif self.scheduler == "cosine":
            sched_dict = self.configure_scheduler_cosine(
                optimizer=opt, lr_ratio=self.lr_ratio, min_lr=self.lr / self.lr_ratio
            )
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

            sched_dict = self.configure_scheduler_oncecycle(
                optimizer=opt,
                lr_ratio=self.lr_ratio,
                learning_rate=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                accumulate_grad_batches=self.trainer.accumulate_grad_batches,
                max_epochs=self.trainer.max_epochs,
            )

        else:
            raise ValueError(
                "Scheduler should be one of 'plateau' or 'cosine', passed: ",
                self.scheduler,
            )
        # TODO check if using different optimizers for different parts of the
        # model would work better
        logger.info(f"\n\n>>> Setting up schedulers:\n\n{sched_dict}")

        return [opt], [sched_dict]

    def plot_scheduler_lr(self):
        """
        Plot the learning rate of the scheduler.

        This is useful to see how the learning rate changes during training,
        and to make sure that the scheduler is working as intended.

        """
        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = 1000
        try:
            accumulate_grad_batches = self.trainer.accumulate_grad_batches
        except RuntimeError:
            accumulate_grad_batches = 1
        spe = steps_per_epoch // accumulate_grad_batches

        optimizer, schedulers = self.configure_optimizers()
        optimizer = optimizer[0]
        scheduler = schedulers[0]["scheduler"]

        xs = list(range(spe))
        lrs = []
        for i in xs:
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        uniplot.plot(np.log1p(np.array(lrs)), xs, title="Learning Rate Schedule")

    def _step(self, batch: TrainBatch, batch_idx: int) -> dict[str, Tensor]:
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
            (0, self.NUM_FRAGMENT_EMBEDDINGS - batch.spectra.size(1)),
            mode="constant",
        )

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
            logger.warning(
                f"Skipping addition of irt loss on batch {batch_idx} "
                f"with value {loss_irt.flatten()}, "
                f"preds: {yhat_irt.flatten()}"
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

            logger.error(
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
                logger.error(self.cpu().state_dict())

                logger.error("last succesfull state:")
                logger.error(self.last_state)
                raise RuntimeError(
                    "Too many nan in a row, dumping state to 'broken_state.pt'"
                )

        else:
            self.num_failed = 0
            if batch_idx % 50 == 0:
                logger.debug("Saved state dict")
                self.last_state = copy.deepcopy(self.state_dict())

        return losses

    def training_step(self, batch: TrainBatch, batch_idx: int | None = None) -> Tensor:
        """See pytorch_lightning documentation."""
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"train_" + k: v for k, v in step_out.items()}
        log_dict.update({"LR": self.trainer.optimizers[0].param_groups[0]["lr"]})

        self.log_dict(
            log_dict,
            prog_bar=True,
            # reduce_fx=nanmean,
        )

        return step_out["l"]

    def on_train_start(self) -> None:
        logger.info("Weights before the start of the training epoch:")
        self.last_state = copy.deepcopy(self.state_dict())
        logger.info(self.last_state)
        return super().on_train_start()

    def validation_step(
        self, batch: TrainBatch, batch_idx: int | None = None
    ) -> Tensor:
        """See pytorch_lightning documentation."""
        step_out = self._step(batch, batch_idx=batch_idx)

        self.irt_metric.update(step_out["irt_l"])
        self.loss_metric.update(step_out["l"])
        self.spectra_metric.update(step_out["spec_l"])
        self.spectra_metric2.update(step_out["spec_l2"])

        return step_out["l"]

    def validation_epoch_end(self, outputs: list[Tensor]) -> list[Tensor]:
        """See pytorch lightning documentation."""
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
        batch_idx: int | None,
    ) -> dict[str, Tensor]:
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
        self, batch, batch_idx: int | None = None
    ) -> tuple[dict[str, Tensor], PredictionResults]:
        losses, pred_out = self._evaluation_step(batch=batch, batch_idx=batch_idx)
        return losses, pred_out.irt, batch.irt

    def test_epoch_end(self, results: list):
        self.metric_calculator.trainer = self.trainer
        self.metric_calculator.log_dict = self.log_dict
        return self.metric_calculator.test_epoch_end(results)

    def predict_step(self, batch: TrainBatch, batch_idx: int | None = None):
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
                                    logger.error(
                                        f"nan mean gradient for {name}: {param.grad}"
                                    )
                                self.log(name, val, prog_bar=True, on_step=True)
                        except AttributeError:
                            msg.append(name)
                        except ValueError:
                            msg.append(name)

        if len(msg) > 0:
            logger.warning(
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
    """Checks the prediction of the model on the iRT peptides.

    Predicts all the procal and Biognosys iRT peptides and checks the correlation
    of the theoretical iRT values and the predicted ones

    Parameters:
        model: PepTransformerModel
            A model to test the predictions on
    """
    model.eval()
    real_rt = []
    pred_rt = []
    for seq, desc in IRT_PEPTIDES.items():
        with torch.no_grad():
            out = model.predict_from_seq(f"{seq}/2", 25)
            pred_rt.append(out.irt.clone().cpu().numpy())
            real_rt.append(np.array(desc["irt"]))

    fit = polyfit(np.array(real_rt).flatten(), np.array(pred_rt).flatten())
    logger.info(fit)
    uniplot.plot(xs=np.array(real_rt).flatten(), ys=np.array(pred_rt).flatten())
    return fit
