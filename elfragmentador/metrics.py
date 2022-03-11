import logging
from math import pi as PI
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import uniplot
from torch import Tensor, nn

from elfragmentador import constants
from elfragmentador.named_batches import EvaluationLossBatch, PredictionResults
from elfragmentador.utils_data import cat_collate


class CosineLoss(torch.nn.CosineSimilarity):
    """
    CosineLoss Implements a simple cosine similarity based loss.
    """

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
            >>> loss(torch.ones([1,2,5]), torch.zeros([1,2,5]))
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss(torch.ones([1,2,5]), 5*torch.ones([1,2,5]))
            tensor([[0., 0., 0., 0., 0.]])
            >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss = CosineLoss(dim=2, eps=1e-4)
            >>> x = [[[0.1, 0.2, 1],[1, 0.2, 0.1]]]
            >>> y = [[[0.2, 0.3, 1],[1, 0.2, 0.1]]]
            >>> torch.tensor(x).shape
            torch.Size([1, 2, 3])
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.0085, 0.0000]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 13.0]]]
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.0000, 0.4909]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 0.0]]]
            >>> # The first tensor is a scaled version, and the second
            >>> # has a missmatch
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.000, 0.1021]])
        """
        out = super().forward(truth, prediction)
        out = 1 - out
        return out


class SpectralAngle(torch.nn.CosineSimilarity):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__(dim=dim, eps=eps)

    def forward(self, truth, prediction):
        """
        Forward calculates the loss.

        Parameters:
            truth : Tensor
            prediction : Tensor

        Returns:
            Tensor

        Examples:
            >>> loss = SpectralAngle(dim=1, eps=1e-4)
            >>> loss(torch.ones([1,2,5]), torch.zeros([1,2,5]))
            tensor([[0., 0., 0., 0., 0.]])
            >>> loss(torch.ones([1,2,5]), 5*torch.ones([1,2,5]))
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
            tensor([[0., 0., 0., 0., 0.]])
            >>> loss = SpectralAngle(dim=2, eps=1e-4)
            >>> x = [[[0.1, 0.2, 1],[1, 0.2, 0.1]]]
            >>> y = [[[0.2, 0.3, 1],[1, 0.2, 0.1]]]
            >>> torch.tensor(x).shape
            torch.Size([1, 2, 3])
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.9169, 1.0000]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.1]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 0.1]]]
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[1.000, 1.000]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.0]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 0.1]]]
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[1.000, 0.9378]])
        """
        out = super().forward(truth, prediction)
        out = 2 * (torch.acos(out) / PI)
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
            >>> loss(torch.ones([1,2,5]), torch.zeros([1,2,5]))
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss(torch.ones([1,2,5]), 5*torch.ones([1,2,5]))
            tensor([[0., 0., 0., 0., 0.]])
            >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss = SpectralAngleLoss(dim=2, eps=1e-4)
            >>> x = [[[0.1, 0.2, 1],[1, 0.2, 0.1]]]
            >>> y = [[[0.2, 0.3, 1],[1, 0.2, 0.1]]]
            >>> torch.tensor(x).shape
            torch.Size([1, 2, 3])
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.0831, 0.0000]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.1]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 0.0]]]
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.0000, 0.0622]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 0.0]]]
            >>> # The first tensor is a scaled version, and the second
            >>> # has a missmatch
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.000, 0.2902]])
        """
        return 1 - super().forward(truth, prediction)


class PearsonCorrelation(torch.nn.Module):
    """
    PearsonCorrelation Implements a simple pearson correlation.
    """

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
        denom_1 = torch.sqrt(torch.sum(vx ** 2, axis=self.axis))
        denom_2 = torch.sqrt(torch.sum(vy ** 2, axis=self.axis))
        denom = (denom_1 * denom_2) + self.eps
        cost = num / denom
        return cost


class MetricCalculator(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.mse_loss = nn.MSELoss(reduction="none")
        self.cosine_loss = CosineLoss(dim=1, eps=1e-8)
        self.angle_loss = SpectralAngleLoss(dim=1, eps=1e-8)

    def forward(self, pred: PredictionResults, gt: PredictionResults):
        return self.calculate_metrics(pred=pred, gt=gt)

    def calculate_metrics(self, pred: PredictionResults, gt: PredictionResults):

        yhat_irt, yhat_spectra = pred.irt.float(), F.normalize(
            torch.relu(self.pad_spectra(pred.spectra)), 2, 1
        )
        irt, spectra = gt.irt.float(), F.normalize(
            torch.relu(self.pad_spectra(gt.spectra)), 2, 1
        )

        loss_irt = self.mse_loss(yhat_irt, irt)
        loss_angle = self.angle_loss(yhat_spectra, spectra)
        loss_cosine = self.cosine_loss(yhat_spectra, spectra)

        return loss_irt.squeeze(), loss_angle.squeeze(), loss_cosine.squeeze()

    @staticmethod
    def pad_spectra(spec):
        spec = F.pad(
            spec, (0, constants.NUM_FRAG_EMBEDINGS - spec.size(1)), mode="constant"
        )
        return spec

    def test_step(self, batch: Dict[str, PredictionResults], batch_idx: int):
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
