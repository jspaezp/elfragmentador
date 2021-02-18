import time
from typing import Dict, List, Tuple, Union
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
import uniplot

class PearsonCorrelation(torch.nn.Module):
    """PearsonCorrelation Implements a simple pearson correlation."""
    def __init__(self, axis=1, eps = 1e-4):
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
        >>> pl.seed_everything(42)
        42
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
        vx = x - torch.mean(x, axis = self.axis).unsqueeze(self.axis)
        vy = y - torch.mean(y, axis = self.axis).unsqueeze(self.axis)

        num = torch.sum(vx * vy, axis = self.axis)
        denom_1 = torch.sqrt(torch.sum(vx ** 2, axis = self.axis))
        denom_2 = torch.sqrt(torch.sum(vy ** 2, axis = self.axis))
        denom = (denom_1 * denom_2) + self.eps
        cost =  num / denom
        return cost

def build_evaluate_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "checkpoint_path", type=str, help="Checkpoint to use for the testing"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sptxt", type=str, help="Sptxt file to use for testing")
    input_group.add_argument("--csv", type=str, help="Sptxt file to use for testing")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to move the model to during the evaluation",
    )
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
    print(summ_out)
    if out_csv is not None:
        print(f">>> Saving results to {out_csv}")
        out.to_csv(out_csv, index=False)


def evaluate_on_sptxt(model, filepath, batch_size=4, device="cpu", max_spec=1e6, *args, **kwargs):
    ds = PeptideDataset.from_sptxt(filepath=filepath, max_spec=max_spec, *args, **kwargs)
    return evaluate_on_dataset(
        model=model, dataset=ds, batch_size=batch_size, device=device
    )


def evaluate_on_csv(model, filepath, batch_size=4, device="cpu", max_spec=1e6):
    ds = PeptideDataset.from_csv(filepath=filepath, max_spec=max_spec)
    return evaluate_on_dataset(
        model=model, dataset=ds, batch_size=batch_size, device=device
    )

def terminal_plot_similarity(similarities, name = ""):
    if all([np.isnan(x) for x in similarities]):
        print("Skipping because all values are missing")
        return None

    uniplot.histogram(
        similarities,
        title=f"{name} mean:{similarities.mean()}",
    )

    qs = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    similarity_quantiles = np.quantile(1-similarities, qs)
    p90 = similarity_quantiles[2]
    p10 = similarity_quantiles[-3]
    q1 = similarity_quantiles[5]
    med = similarity_quantiles[4]
    q3 = similarity_quantiles[3]
    title = f"Accumulative distribution (y) of the 1 - {name} (x)"
    title += f"\nP90={1-p90:.3f} Q3={1-q3:.3f}"
    title += f" Median={1-med:.3f} Q1={1-q1:.3f} P10={1-p10:.3f}"
    uniplot.plot(
        xs = similarity_quantiles,
        ys = qs,
        lines=True,
        title=title)

def evaluate_on_dataset(
    model: PepTransformerModel,
    dataset: PeptideDataset,
    batch_size: int = 4,
    device: str = "cpu",
    overwrite_nce: Union[float, bool] = False,
) -> Tuple[Dict[str, Union[Series, ndarray]], Dict[str, Union[float64, float32]]]:
    dl = torch.utils.data.DataLoader(dataset, batch_size)
    cs = torch.nn.CosineSimilarity()
    pc = PearsonCorrelation()

    model.eval()
    model.to(device)
    rt_results = []
    mod_sequences = dataset.df["ModSequences"]
    rt_real = dataset.df["RTs"]
    irt_real = dataset.df["iRT"]

    if sum(~np.isnan(np.array(irt_real).astype("float"))) > 1:
        print("Using iRT instead of RT")
        rt_real = irt_real

    charges = dataset.df["Charges"]
    spec_results_cs = []
    spec_results_pc = []

    print(">>> Starting Evaluation of the spectra <<<")
    start_time = time.time()
    with torch.no_grad():
        for b in tqdm(dl):
            if overwrite_nce:
                nce = torch.where(
                    torch.tensor(True),
                    torch.tensor(overwrite_nce),
                    b.nce)
            else:
                nce = b.nce

            outs = model.forward(
                src=b.encoded_sequence.clone().to(device),
                charge=b.charge.clone().to(device),
                mods=b.encoded_mods.clone().to(device),
                nce=nce.clone().to(device),
            )

            out_spec = outs.spectra.cpu().clone()
            out_spec = out_spec / out_spec.max(axis=1).values.unsqueeze(0).T

            spec_results_cs.append(cs(out_spec, b.encoded_spectra))
            spec_results_pc.append(pc(out_spec, b.encoded_spectra))
            rt_results.append(outs.irt.cpu().clone().flatten())
            del b
            del outs

    end_time = time.time()
    elapsed_time = end_time - start_time

    rt_results = torch.cat(rt_results) * 100
    spec_results_pc = torch.cat(spec_results_pc)
    spec_results_cs = torch.cat(spec_results_cs)

    print(f">> Elapsed time for {len(spec_results_cs)} results was {elapsed_time}.")
    print(
        f">> {len(spec_results_cs) / elapsed_time} results/sec"
        f"; {elapsed_time / len(spec_results_cs)} sec/res"
    )
    out = {
        "ModSequence": mod_sequences,
        "Charges": charges,
        "Predicted_iRT": rt_results.numpy().flatten(),
        "Real_RT": rt_real.to_numpy().flatten(),
        "Spectra_Similarity_Cosine": spec_results_cs.numpy().flatten(),
        "Spectra_Similarity_Pearson": spec_results_pc.numpy().flatten(),
    }

    terminal_plot_similarity(out['Spectra_Similarity_Pearson'], "Pearson Similarity")
    terminal_plot_similarity(out['Spectra_Similarity_Cosine'], "Cosine Similarity")

    # TODO consider the possibility of stratifying on files before normalizing
    missing_vals = np.isnan(np.array(rt_real).astype("float"))
    print(
        f"Will remove {sum(missing_vals)}/{len(missing_vals)} "
        "because they have missing iRTs"
    )
    norm_p_irt, rev_p_irt = norm(out["Predicted_iRT"])
    norm_r_irt, rev_r_irt = norm(out["Real_RT"])

    if sum(missing_vals) == len(norm_p_irt):
        rt_fit = {'determination': None}
    else:
        rt_fit = polyfit(norm_p_irt[~missing_vals], norm_r_irt[~missing_vals])

        uniplot.plot(
            ys=rev_p_irt(norm_p_irt)[~missing_vals],
            xs=rev_r_irt(norm_r_irt)[~missing_vals],
            title=(
                f"Predicted iRT (y) vs RT (x)"
                f" (normalized R2={rt_fit['determination']})"
            ),
        )

        rt_errors = abs(rev_r_irt(norm_r_irt) - rev_r_irt(norm_p_irt))
        out.update({"RT_Error": rt_errors})
        terminal_plot_similarity(rt_errors[~missing_vals], "RT prediction error")

    summ_out = {
        "normRT Rsquared": rt_fit["determination"],
        "AverageSpectraCosineSimilarity": out["Spectra_Similarity_Cosine"].mean(),
        "AverageSpectraPearsonSimilarty": out["Spectra_Similarity_Pearson"].mean(),
    }
    return pd.DataFrame(out), summ_out


def norm(x: ndarray) -> ndarray:
    """Normalizes a numpy array by substracting mean and dividing by standard deviation"""
    sd = np.nanstd(x)
    m = np.nanmean(x)
    out = (x - m) / sd
    return out, lambda y: (y*sd) + m

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

def evaluate_landmark_rt(model: PepTransformerModel):
    model.eval()
    real_rt = []
    pred_rt = []
    for seq, desc in constants.IRT_PEPTIDES.items():
        with torch.no_grad():
            out = model.predict_from_seq(seq, 2, 25)
            pred_rt.append(100*out.irt.numpy())
            real_rt.append(np.array(desc['irt']))

    uniplot.plot(
        xs = np.array(real_rt).flatten(),
        ys = np.array(pred_rt).flatten())
