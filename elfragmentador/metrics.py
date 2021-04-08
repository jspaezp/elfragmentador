import torch
from torch import Tensor


class CosineLoss(torch.nn.CosineSimilarity):
    """CosineLoss Implements a simple cosine similarity based loss."""

    def __init__(self, *args, **kwargs) -> None:
        """__init__ Instantiates the class.

        All arguments are passed to `torch.nn.CosineSimilarity`
        """
        super().__init__(*args, **kwargs)

    def forward(self, truth: Tensor, prediction: Tensor) -> Tensor:
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
        >>> loss(torch.ones([1,2,5]), 5*torch.zeros([1,2,5]))
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
