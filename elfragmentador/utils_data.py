from typing import Union, Optional, List, Dict
import warnings
import logging
from torch.utils.data._utils.collate import default_collate

import pandas as pd
from pandas import DataFrame
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tqdm.auto import tqdm

from elfragmentador import constants as CONSTANTS
import uniplot


def collate_fun(batch):
    """Collate function that first equalizes the length of tensors

    Examples:
        >>> collate_fun([torch.ones(2), torch.ones(4), torch.ones(14)])
        tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return _match_lengths(nested_list=batch, verbose=False)
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_fun(samples) for samples in zip(*batch)))

    return default_collate([x for x in batch])


def cat_collate(batch):
    elem = batch[0]
    elemtype = type(elem)

    if isinstance(elem, torch.Tensor):
        return torch.cat(batch)

    if isinstance(elem, dict):
        out = {key: cat_collate([d[key] for d in batch]) for key in elem}
        return out

    out = [torch.cat([y[i] for y in batch]) for i, _ in enumerate(elem)]
    return elemtype(*out)


def _convert_tensor_column(column, elem_function=float, verbose=True, *args, **kwargs):
    """converts a series (column in a pandas dataframe) to a tensor

    Expects a column whose character values actually represent lists of numbers
    for exmaple "[1,2,3,4]"

    Args:
        column: Series/list of stirngs
        elem_function: function to use to convert each substring in a value (Default value = float)
        *args: Arguments to pass to tqdm
        **kwargs: Keywords arguments passed to tqdm

    Returns:
        List: A nested list containing the converted values

    Examples:
        >>> _convert_tensor_column(["[2,2]", "[3,3]"], elem_function=float)
        [array([2., 2.]), array([3., 3.])]
    """

    # next(col.__iter__()) is the equivalent of col[0] but works for some
    # series where the index 0 does not exist
    if isinstance(next(iter(column)), str):
        out = [
            np.array(
                [
                    elem_function(y.strip())
                    for y in x.strip("[]").replace("'", "").split(",")
                ]
            )
            for x in tqdm(column, disable=not verbose, *args, **kwargs)
        ]

    elif isinstance(next(iter(column)), list) or isinstance(next(iter(column)), tuple):
        out = [
            np.array([elem_function(y) for y in x])
            for x in tqdm(column, disable=not verbose, *args, **kwargs)
        ]

    else:
        logging.warning("Passed column is not a string, skipping conversion")
        out = column

    return out


def _match_lengths(
    nested_list: Union[List[List[Union[int, float]]], List[List[int]]],
    max_len: Optional[int] = None,
    name: str = "items",
    verbose=True,
) -> Tensor:
    """
    match_lengths Matches the lengths of all tensors in a list

    Args:
        nested_list (List[np.ndarray]):
            A list of numpy arrays
        max_len (int, optional): Length to match all tensors to, if not provided will pad to the max found
        name (str, optional): name to use (just for logging purposes). Defaults to "items".
        verbose (bool): Wether to log the matching criteria, defaults to False

    Returns:
        Tensor:
            Tensor product of stacking all the elements in the input nested list, after
            equalizing the length of all of them to the specified max_len

    Examples:
        >>> out = _match_lengths([np.array([3, 3]), np.array([1, 2])], 2, "")
        >>> out
        tensor([[3, 3],
                [1, 2]])
        >>> out[0]
        tensor([3, 3])
        >>> _match_lengths([np.array([1]), np.array([1, 2])], 2, "")
        tensor([[1, 0],
                [1, 2]])
        >>> _match_lengths([np.array([1]), np.array([1, 2]), np.array([1,2,3,4,5,6])])
        tensor([[1, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6]])
    """
    elem = nested_list[0]
    if isinstance(elem, torch.Tensor):
        pad_fun = F.pad
        conversion_fun = lambda x: x
    else:
        pad_fun = np.pad
        conversion_fun = torch.from_numpy

    lengths = np.array([len(x) for x in nested_list])
    unique_lengths = set(lengths)
    max_found = max(unique_lengths)
    if max_len is None:
        max_len = max_found

    match_max = lengths == max_len

    out_message = (
        f"{len(match_max)}/{len(nested_list)} "
        f"{name} actually match the max sequence length of"
        f" {max_len},"
        f" found {unique_lengths}"
    )

    if (len(unique_lengths) == 1) and (max_len in unique_lengths):
        if verbose:
            logging.info(out_message)
        out = torch.stack([conversion_fun(x) for x in nested_list], dim=0)
    else:
        if verbose:
            logging.warning(out_message)
        out = [
            pad_fun(x, (0, max_len - len(x)), "constant") if len(x) != max_len else x
            for x in nested_list
        ]
        out = torch.stack([conversion_fun(x) for x in out], dim=0)

    return out


def _match_colnames(df: DataFrame) -> Dict[str, Optional[str]]:
    """
    match_colnames Tries to find aliases for columns names in a data frame


    Tries to find the following column aliases:

    "SeqE": Sequence encoding
    "ModE": Modification Encoding
    "SpecE": Spectrum encoding
    "Ch": Charge
    "iRT": Retention time
    "NCE": Collision Energy
    "Weight": Weight of each spectrum (will default to use Weight or reps columns)


    Args:
        df (DataFrame): Data frame to find columns for ...

    Returns:
        Dict[str, Optional[str]]:
            Dictionary with the aliases (keys are the ones specified in the details section)
    """

    def _match_col(string1, string2, colnames, match_mode="in", combine_mode=None):
        m = {
            "in": lambda q, t: q in t,
            "startswith": lambda q, t: q.startswith(t) or t.startswith(q),
            "equals": lambda q, t: q == t,
        }
        match_fun = m[match_mode]
        match_indices1 = [i for i, x in enumerate(colnames) if match_fun(string1, x)]

        if string2 is None:
            match_indices = match_indices1
        else:
            match_indices2 = [
                i for i, x in enumerate(colnames) if match_fun(string2, x)
            ]
            if combine_mode == "union":
                match_indices = set(match_indices1).union(set(match_indices2))
            elif combine_mode == "intersect":
                match_indices = set(match_indices1).intersection(set(match_indices2))
            else:
                raise NotImplementedError

        try:
            out_index = list(match_indices)[0]
        except IndexError:
            out_index = None

        return out_index

    colnames = list(df)
    out = {
        "SeqE": _match_col("Encoding", "Seq", colnames, combine_mode="intersect"),
        "ModE": _match_col("Encoding", "Mod", colnames, combine_mode="intersect"),
        "SpecE": _match_col("Encoding", "Spec", colnames, combine_mode="intersect"),
        "Ch": _match_col("harg", None, colnames),
        "iRT": _match_col("IRT", "iRT", colnames, combine_mode="union"),
        "NCE": _match_col(
            "nce", "NCE", colnames, combine_mode="union", match_mode="startswith"
        ),
        "Weight": _match_col(
            "Weight", "reps", colnames, match_mode="in", combine_mode="union"
        ),
    }
    out = {k: (colnames[v] if v is not None else None) for k, v in out.items()}
    logging.info(f">>> Mapped column names to the provided dataset {out}")
    return out


def _convert_tensor_columns_df(df, verbose=True):
    name_match = _match_colnames(df)

    parsable_cols = [("SeqE", int), ("ModE", int), ("SpecE", float)]

    for col, fun in parsable_cols:
        df[name_match[col]] = _convert_tensor_column(
            df[name_match[col]], fun, verbose=verbose
        )

    return df


def _filter_df_on_sequences(df: DataFrame, name: str = "") -> DataFrame:
    """
    filter_df_on_sequences Filters a DataFrame for the peptides that correctly match the expected lengths

    Args:
        df (DataFrame): A DataFrame to filter
        name (str, optional): Only used for debugging purposes. Defaults to "".

    Returns:
        DataFrame: The filtered dataframe
    """
    name_match = _match_colnames(df)
    logging.info(list(df))
    logging.warning(f"Removing Large sequences, currently {name}: {len(df)}")

    seq_iterable = _convert_tensor_column(
        df[name_match["SeqE"]], int, "Decoding tensor seqs"
    )
    df[name_match["SeqE"]] = seq_iterable
    seq_len_matching = [len(x) <= CONSTANTS.MAX_TENSOR_SEQUENCE for x in seq_iterable]
    if sum(seq_len_matching) == 0:
        warnings.warn("No sequences have the expected length")

    df = df[seq_len_matching].copy().reset_index(drop=True)

    logging.warning(f"Left {name}: {len(df)}")
    return df


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
