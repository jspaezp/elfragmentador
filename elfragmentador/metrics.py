import torch
from torch import Tensor
from math import pi as PI


class CosineLoss(torch.nn.CosineSimilarity):
    """CosineLoss Implements a simple cosine similarity based loss."""

    def __init__(self, *args, **kwargs) -> None:
        """__init__ Instantiates the class.

        All arguments are passed to `torch.nn.CosineSimilarity`
        """
        super().__init__(*args, **kwargs)

    def forward(self, truth: Tensor, prediction: Tensor) -> Tensor:
        """Forward calculates the loss.

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
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
            tensor([[0., 0., 0., 0., 0.]])
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, truth, prediction):
        """Forward calculates the loss.

        Parameters:
            truth : Tensor
            prediction : Tensor

        Returns:
            Tensor

        Examples:
            >>> loss = SpectralAngle(dim=1, eps=1e-4)
            >>> loss(torch.ones([1,2,5]), torch.zeros([1,2,5]))
            tensor([[1., 1., 1., 1., 1.]])
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
        """Forward calculates the loss.

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
            tensor([[1., 1., 1., 1., 1.]])
            >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
            tensor([[0., 0., 0., 0., 0.]])
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
            tensor([[0., 0.]])
            >>> x = [[[0.2, 0.4, 2],[1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1],[1, 0.2, 0.0]]]
            >>> # The first tensor is a scaled version, and the second
            >>> # has a missmatch
            >>> loss(torch.tensor(x), torch.tensor(y))
            tensor([[0.000, 0.2902]])
        """
        return 1 - super().forward(truth, prediction)


class PearsonCorrelation(torch.nn.Module):
    """PearsonCorrelation Implements a simple pearson correlation."""

    def __init__(self, axis=1, eps=1e-4):
        """__init__ Instantiates the class.

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
        """Forward calculates the loss.

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
        >>> loss(torch.ones([1,2,5]), torch.zeros([1,2,5]))
        tensor([[1., 1., 1., 1., 1.]])
        >>> loss(torch.ones([1,2,5]), 5*torch.ones([1,2,5]))
        tensor([[1., 1., 1., 1., 1.]])
        >>> loss(torch.zeros([1,2,5]), torch.zeros([1,2,5]))
        tensor([[0., 0., 0., 0., 0.]])
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
