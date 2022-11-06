import logging
from math import pi as PI

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import uniplot
from torch import Tensor, nn

from elfragmentador.config import get_default_config
from elfragmentador.named_batches import EvaluationLossBatch, PredictionResults
from elfragmentador.utils_data import cat_collate

DEFAULT_CONFIG = get_default_config()


class CosineLoss(torch.nn.CosineSimilarity):
    """CosineLoss Implements a simple cosine similarity based loss."""

    def __init__(self, dim=1, eps=1e-8) -> None:
        """
        __init__ Instantiates the class.

        All arguments are passed to `torch.nn.CosineSimilarity`
        """
        super().__init__(dim=dim, eps=eps)

    def forward(self, truth: Tensor, prediction: Tensor) -> Tensor:
        """
        Forward calculates the loss.

        Parameters:
            truth : Tensor
            prediction : Tensor

        Returns:
            Tensor

        Examples:
            >>> loss = CosineLoss(dim=1, eps=1e-4)

            >>> x = torch.ones([1,2,5])
            >>> y = torch.zeros([1,2,5]) + 0.1
            >>> calc_loss = loss(x, y)
            >>> calc_loss.round(decimals = 2)
            tensor([[0., 0., 0., 0., 0.]])
        """
        out = super().forward(truth, prediction)
        out = 1 - out
        return out


class SpectralAngle(torch.nn.CosineSimilarity):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__(dim=dim, eps=eps)

    def forward(self, truth, prediction):
        """
        Forward calculates the similarity.

        Parameters:
            truth : Tensor
            prediction : Tensor

        Returns:
            Tensor

        Examples:
            >>> loss = SpectralAngle(dim=1, eps=1e-4)

            >>> x = torch.ones([1,2,5])
            >>> y = torch.zeros([1,2,5]) + 0.1
            >>> calc_loss = loss(x, y)
            >>> calc_loss.round(decimals = 2)
            tensor([[1., 1., 1., 1., 1.]])
        """
        out = super().forward(truth, prediction)

        # Here clamp is needed to avoid nan values
        # Where due to numerical errors the cosine similarity
        # is slightly larger than 1
        out = 2 * (torch.acos(out.clamp(0, 1)) / PI)
        out = 1 - out

        return out


class SpectralAngleLoss(SpectralAngle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, truth, prediction):
        """
        Forward calculates the loss.

        Parameters:
            truth : Tensor
            prediction : Tensor

        Returns:
            Tensor

        Examples:
            >>> loss = SpectralAngleLoss(dim=1, eps=1e-4)

            >>> x = torch.ones([1,2,5])
            >>> y = torch.zeros([1,2,5]) + 0.1
            >>> calc_loss = loss(x, y)
            >>> calc_loss.round(decimals = 2)
            tensor([[0., 0., 0., 0., 0.]])
        """
        return 1 - super().forward(truth, prediction)


class PearsonCorrelation(torch.nn.Module):
    """PearsonCorrelation Implements a simple pearson correlation."""

    def __init__(self, axis=1, eps=1e-4):
        """
        __init__ Instantiates the class.

        Creates a callable object to calculate the pearson correlation on an axis

        Parameters
        ----------
        axis : int, optional
            The axis over which the correlation is calculated.
            For instance, if the input has shape [5, 500] and the axis is set
            to 1, the output will be of shape [5]. On the other hand, if the axis
            is set to 0, the output will have shape [500], by default 1
        eps : float, optional
            Number to be added to to prevent division by 0, by default 1e-4
        """
        super().__init__()
        self.axis = axis
        self.eps = eps

    def forward(self, x, y):
        """
        Forward calculates the loss.

        Parameters
        ----------
        truth : Tensor
        prediction : Tensor

        Returns
        -------
        Tensor

        Examples
        --------
        >>> loss = PearsonCorrelation(axis=1, eps=1e-4)
        >>> loss(torch.tensor([[1.,2.,3.],[4.,5.,6.]]),
        ... torch.tensor([[1.1,2.0,3.2],[4.1,5.0,6.2]]))
        tensor([0.9966, 0.9966])
        >>> out = loss(torch.rand([5, 174]), torch.rand([5, 174]))
        >>> out.shape
        torch.Size([5])
        >>> loss = PearsonCorrelation(axis=0, eps=1e-4)
        >>> out = loss(torch.rand([5, 174]), torch.rand([5, 174]))
        >>> out.shape
        torch.Size([174])
        """
        vx = x - torch.mean(x, axis=self.axis).unsqueeze(self.axis)
        vy = y - torch.mean(y, axis=self.axis).unsqueeze(self.axis)

        num = torch.sum(vx * vy, axis=self.axis)
        denom_1 = torch.sqrt(torch.sum(vx**2, axis=self.axis))
        denom_2 = torch.sqrt(torch.sum(vy**2, axis=self.axis))
        denom = (denom_1 * denom_2) + self.eps
        cost = num / denom
        return cost


class MetricCalculator(pl.LightningModule):
    config = DEFAULT_CONFIG

    def __init__(self) -> None:
        """Implements a dummy nn.Module to calculate metrics."""

        super().__init__()

        self.mse_loss = nn.MSELoss(reduction="none")
        self.cosine_loss = CosineLoss(dim=1, eps=1e-8)
        self.angle_loss = SpectralAngleLoss(dim=1, eps=1e-8)

    def forward(self, pred: PredictionResults, gt: PredictionResults):
        return self.calculate_metrics(pred=pred, gt=gt)

    def calculate_metrics(self, pred: PredictionResults, gt: PredictionResults):
        yhat_spectra = self.pad_spectra(
            pred.spectra, self.config.num_fragment_embeddings
        )
        yhat_irt = pred.irt.float()
        yhat_spectra = F.normalize(
            torch.relu(yhat_spectra),
            2,
            1,
        )
        spectra = self.pad_spectra(gt.spectra, self.config.num_fragment_embeddings)
        irt = gt.irt.float()
        spectra = F.normalize(
            torch.relu(spectra),
            2,
            1,
        )

        loss_irt = self.mse_loss(yhat_irt, irt)
        loss_angle = self.angle_loss(yhat_spectra, spectra)
        loss_cosine = self.cosine_loss(yhat_spectra, spectra)

        return loss_irt, loss_angle, loss_cosine

    @staticmethod
    def pad_spectra(spec, num_frag_embeddings):
        spec = F.pad(spec, (0, num_frag_embeddings - spec.size(1)), mode="constant")
        return spec

    def test_step(self, batch: dict[str, PredictionResults], batch_idx: int):
        loss_irt, loss_angle, loss_cosine = self.calculate_metrics(
            gt=batch["gt"], pred=batch["pred"]
        )
        losses = {
            "loss_irt": loss_irt,
            "loss_angle": loss_angle,
            "loss_cosine": loss_cosine,
        }
        return losses, batch["pred"].irt, batch["gt"].irt

    def test_epoch_end(self, outputs):
        losses = cat_collate([x[0] for x in outputs])
        pred_irt_outs = torch.cat([x[1] for x in outputs])
        truth_irt = torch.cat([x[2] for x in outputs])

        nm_pred_irt_outs = pred_irt_outs[~torch.isnan(truth_irt)]
        nm_truth_irt = truth_irt[~torch.isnan(truth_irt)]

        norm_pred_irt = (
            pred_irt_outs - nm_pred_irt_outs.mean()
        ) / nm_pred_irt_outs.std()
        norm_truth_irt = (truth_irt - nm_truth_irt.mean()) / nm_truth_irt.std()

        if hasattr(self.trainer, "plot") and self.trainer.plot:
            xs = norm_pred_irt[~torch.isnan(norm_truth_irt)].cpu().detach().numpy()
            ys = norm_truth_irt[~torch.isnan(norm_truth_irt)].cpu().detach().numpy()

            if len(xs) > 0:
                try:
                    uniplot.plot(
                        xs=xs,
                        ys=ys,
                        title="Scaled ground truth (y) vs scaled prediction(x) of RT",
                    )
                except AssertionError as e:
                    logging.error(f"Failed to generate plot with error {e}")
            else:
                logging.error(
                    "All values are missing for retention time, skipping plotting"
                )

        losses.update({"scaled_se_loss": self.mse_loss(norm_pred_irt, norm_truth_irt)})
        self.log_dict({"median_" + k: v.median() for k, v in losses.items()})

        self.trainer.test_results = EvaluationLossBatch(**losses)
