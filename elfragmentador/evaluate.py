from elfragmentador.predictor import Predictor
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import pytorch_lightning as pl
import numpy as np
from numpy import float32, float64, ndarray

import pandas as pd
from pandas.core.series import Series
from tqdm.auto import tqdm

from elfragmentador import constants
from elfragmentador.model import PepTransformerModel
from elfragmentador.datamodules import PeptideDataset
from elfragmentador.metrics import PearsonCorrelation
from elfragmentador.math_utils import norm, polyfit
import uniplot


def build_evaluate_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sptxt", type=str, help="Sptxt file to use for testing")
    input_group.add_argument("--csv", type=str, help="Sptxt file to use for testing")
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size to use during the valuation",
    )
    nce_group = parser.add_mutually_exclusive_group()
    nce_group.add_argument(
        "--overwrite_nce",
        type=float,
        help="NCE to overwrite the collision energy with",
    )
    nce_group.add_argument(
        "--screen_nce",
        type=str,
        help="Comma delimited series of collision energies to use",
    )
    parser.add_argument(
        "--max_spec",
        default=1e6,
        type=int,
        help="Maximum number of spectra to read",
    )
    parser.add_argument(
        "--out_csv", type=str, help="Optional csv file to output results to"
    )
    return parser


# Given a model checkpoint and some input data, parse the data and return metrics, also a csv with the report
def evaluate_checkpoint(
    checkpoint_path: str,
    sptxt_path: str,
    batch_size=4,
    device="cpu",
    out_csv=None,
    max_spec=1e6,
):
    model = PepTransformerModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()

    out, summ_out = evaluate_on_sptxt(
        model,
        filepath=sptxt_path,
        batch_size=batch_size,
        device=device,
        max_spec=max_spec,
    )
    out = pd.DataFrame(out).sort_values(["Spectra_Similarity"]).reset_index()
    logging.info(summ_out)
    if out_csv is not None:
        logging.info(f">>> Saving results to {out_csv}")
        out.to_csv(out_csv, index=False)


def evaluate_on_sptxt(
    model: PepTransformerModel,
    filepath: Union[Path, str],
    batch_size=4,
    device="cpu",
    max_spec=1e6,
    *args,
    **kwargs,
):
    ds = PeptideDataset.from_sptxt(
        filepath=filepath, max_spec=max_spec, *args, **kwargs
    )
    return evaluate_on_dataset(
        model=model, dataset=ds, batch_size=batch_size, device=device
    )


def evaluate_on_csv(
    model: PepTransformerModel,
    filepath: Union[Path, str],
    batch_size: int = 4,
    device: str = "cpu",
    max_spec: int = 1e6,
):
    ds = PeptideDataset.from_csv(filepath=filepath, max_spec=max_spec)
    return evaluate_on_dataset(
        model=model, dataset=ds, batch_size=batch_size, device=device
    )


# TODO make this a function
def evaluate_on_dataset(
    model: PepTransformerModel,
    dataset: PeptideDataset,
    batch_size: int = 4,
    gpus: int = 0,
    overwrite_nce: Optional[float] = False,
) -> Tuple[pd.DataFrame, Dict[str, Union[float64, float32]]]:
    predictor = Predictor(gpus=gpus)

    outs = predictor.evaluate_dataset(
        model=model, dataset=dataset, batch_size=batch_size
    )

    summ_out = {"median_" + k: v.median() for k, v in outs._asdict().items()}
    return (
        pd.DataFrame({k: v.squeeze().numpy() for k, v in outs._asdict().items()}),
        summ_out,
    )


def evaluate_landmark_rt(model: PepTransformerModel):
    """evaluate_landmark_rt Checks the prediction of the model on the iRT peptides

    Predicts all the procal and Biognosys iRT peptides and checks the correlation
    of the theoretical iRT values and the predicted ones

    Parameters
    ----------
    model : PepTransformerModel
        A model to test the predictions on

    """
    model.eval()
    real_rt = []
    pred_rt = []
    for seq, desc in constants.IRT_PEPTIDES.items():
        with torch.no_grad():
            out = model.predict_from_seq(seq, 2, 25, enforce_length=False)
            pred_rt.append(100 * out.irt.numpy())
            real_rt.append(np.array(desc["irt"]))

    # TODO make this return a correlation coefficient
    fit = polyfit(np.array(real_rt).flatten(), np.array(pred_rt).flatten())
    logging.info(fit)
    uniplot.plot(xs=np.array(real_rt).flatten(), ys=np.array(pred_rt).flatten())
    return fit
