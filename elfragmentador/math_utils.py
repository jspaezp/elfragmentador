import torch
from torchmetrics import Metric


class MissingDataAverager(Metric):
    """
    Metric class that averages values, ignoring missing

    Examples:
    >>> averager = MissingDataAverager()
    >>> averager.update(torch.ones(1))
    >>> averager.update(torch.zeros(1))
    >>> averager.update(torch.tensor([float("nan")]))
    >>> averager.compute()
    tensor(0.5000)
    """

    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, vals: torch.Tensor):
        self.values.append(vals)

    def compute(self):
        # compute final result
        return nanmean(torch.tensor(self.values))


def nanmean(v, *args, inplace=False, **kwargs):
    """
    Function that calculates mean of a tensor while removing missing values

    From: https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
