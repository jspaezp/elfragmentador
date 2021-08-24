import time
import logging
from pathlib import Path
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


def evaluate_on_dataset(
    model: PepTransformerModel,
    dataset: PeptideDataset,
    batch_size: int = 4,
    device: str = "cpu",
    overwrite_nce: Union[float, bool] = False,
) -> Tuple[pd.DataFrame, Dict[str, Union[float64, float32]]]:
    dl = torch.utils.data.DataLoader(dataset, batch_size)
    cs = torch.nn.CosineSimilarity()
    pc = PearsonCorrelation()

    model.eval()
    model.to(device)
    rt_results = []

    spec_results_cs = []
    spec_results_pc = []

    logging.info(">>> Starting Evaluation of the spectra <<<")
    start_time = time.time()
    with torch.no_grad():
        for b in tqdm(dl):
            if overwrite_nce:
                nce = torch.where(
                    torch.tensor(True), torch.tensor(overwrite_nce), b.nce
                )
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

    logging.info(
        f">> Elapsed time for {len(spec_results_cs)} results was {elapsed_time}."
    )
    logging.info(
        f">> {len(spec_results_cs) / elapsed_time} results/sec"
        f"; {elapsed_time / len(spec_results_cs)} sec/res"
    )
    out = {
        "ModSequence": dataset.mod_sequences,
        "Charges": dataset.charges.numpy().flatten(),
        "Predicted_iRT": rt_results.numpy().flatten(),
        "Real_RT": dataset.norm_irts.numpy().astype("float").flatten() * 100,
        "Spectra_Similarity_Cosine": spec_results_cs.numpy().flatten(),
        "Spectra_Similarity_Pearson": spec_results_pc.numpy().flatten(),
    }

    terminal_plot_similarity(out["Spectra_Similarity_Pearson"], "Pearson Similarity")
    terminal_plot_similarity(out["Spectra_Similarity_Cosine"], "Cosine Similarity")

    # TODO consider the possibility of stratifying on files before normalizing
    missing_vals = np.isnan(out["Real_RT"])
    if sum(missing_vals) > 0:
        logging.warning(
            f"Will remove {sum(missing_vals)}/{len(missing_vals)} "
            "because they have missing iRTs"
        )
    norm_p_irt, rev_p_irt = norm(out["Predicted_iRT"])
    norm_r_irt, rev_r_irt = norm(out["Real_RT"])

    if sum(missing_vals) == len(norm_p_irt):
        rt_fit = {"determination": None}
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
            out = model.predict_from_seq(seq, 2, 25)
            pred_rt.append(100 * out.irt.numpy())
            real_rt.append(np.array(desc["irt"]))

    # TODO make this return a correlation coefficient
    fit = polyfit(np.array(real_rt).flatten(), np.array(pred_rt).flatten())
    logging.info(fit)
    uniplot.plot(xs=np.array(real_rt).flatten(), ys=np.array(pred_rt).flatten())
    return fit
