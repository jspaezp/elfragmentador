from elfragmentador.predictor import Predictor
import logging

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


# TODO refactor the code there is really no reason for this file to exist


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
