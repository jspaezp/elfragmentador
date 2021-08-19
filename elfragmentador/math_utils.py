from typing import Union, List, Dict
import torch
from torchmetrics import Metric
from numpy import ndarray, float64
import numpy as np


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


def norm(x: ndarray) -> ndarray:
    """Normalizes a numpy array by substracting mean and dividing by standard deviation"""
    sd = np.nanstd(x)
    m = np.nanmean(x)
    out = (x - m) / sd
    return out, lambda y: (y * sd) + m


# Polynomial Regression
# Implementation from:
# https://stackoverflow.com/questions/893657/
def polyfit(
    x: ndarray, y: ndarray, degree: int = 1
) -> Dict[str, Union[List[float], float64]]:
    """Fits a polynomial fit"""
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results["polynomial"] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results["determination"] = ssreg / sstot

    return results


def apply_polyfit(x, polynomial):
    tmp = 0
    for i, term in enumerate(polynomial[:-1]):
        tmp = tmp + ((x ** (1 + i)) * term)
    tmp = tmp + polynomial[-1]
    return tmp
