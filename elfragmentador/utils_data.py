import logging

import numpy as np
import torch
import uniplot
from torch.utils.data._utils.collate import default_collate


def collate_fun(batch):
    """
    Collate function that first equalizes the length of tensors Modified from.

    the pytorch implementation.

    Examples:
        >>> collate_fun([torch.ones(2), torch.ones(4), torch.ones(14)])
        tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return _match_lengths(nested_list=batch)
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_fun(samples) for samples in zip(*batch)))
    elif isinstance(elem, dict):
        return {key: collate_fun([d[key] for d in batch]) for key in elem}

    return default_collate([x for x in batch])


def cat_collate(batch):
    """
    Collate function that concatenates the first dimension, instead of stacking.

    it.

    Examples:
        >>> collate_fun([torch.ones(4), torch.ones(4), torch.ones(4)])
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]])
    """
    elem = batch[0]
    elemtype = type(elem)

    if isinstance(elem, torch.Tensor):
        return torch.cat([b if len(b.shape) > 0 else b.unsqueeze(0) for b in batch])

    if isinstance(elem, dict):
        out = {key: cat_collate([d[key] for d in batch]) for key in elem}
        return out

    out = [torch.cat([y[i] for y in batch]) for i, _ in enumerate(elem)]
    return elemtype(*out)


def terminal_plot_similarity(similarities, name=""):
    if all([np.isnan(x) for x in similarities]):
        logging.warning("Skipping because all values are missing")
        return None

    uniplot.histogram(
        similarities,
        title=f"{name} mean:{similarities.mean()}",
    )

    qs = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    similarity_quantiles = np.quantile(1 - similarities, qs)
    p90 = similarity_quantiles[2]
    p10 = similarity_quantiles[-3]
    q1 = similarity_quantiles[5]
    med = similarity_quantiles[4]
    q3 = similarity_quantiles[3]
    title = f"Accumulative distribution (y) of the 1 - {name} (x)"
    title += f"\nP90={1-p90:.3f} Q3={1-q3:.3f}"
    title += f" Median={1-med:.3f} Q1={1-q1:.3f} P10={1-p10:.3f}"
    uniplot.plot(xs=similarity_quantiles, ys=qs, lines=True, title=title)
