import logging
from typing import NamedTuple

from pandas import DataFrame


def _log_batches(batches, prefix="Tensor"):
    {logging.debug(f"{prefix}: {k}:{v.shape}") for k, v in batches._asdict().items()}


def _append_batch_to_df(df: DataFrame, batches: NamedTuple, prefix: str):
    logging.info(f"Appending info to dataframe with prefix '{prefix}'")
    {
        logging.debug(f"Appending Batches: {prefix}{k}:{v.shape}")
        for k, v in batches._asdict().items()
    }
    for k, v in batches._asdict().items():
        k = prefix + k
        df.insert(loc=len(list(df)) - 2, column=k, value=float("nan"))
        df[k] = [x.numpy().flatten().tolist() for x in v]
