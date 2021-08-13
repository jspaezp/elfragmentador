import torch


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
