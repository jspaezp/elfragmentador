import time
from tqdm.auto import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import pandas as pd
from elfragmentador.model import PepTransformerModel
from elfragmentador.datamodules import PeptideDataset
from numpy import ndarray


def build_evaluate_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "checkpoint_path", type=str, help="Checkpoint to use for the testing"
    )
    parser.add_argument("sptxt_path", type=str, help="Sptxt file to use for testing")
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
    checkpoint_path: str, sptxt_path: str, batch_size=4, device="cpu", out_csv=None, max_spec=1e6
):
    model = PepTransformerModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()

    out, summ_out = evaluate_on_sptxt(
        model, filepath=sptxt_path, batch_size=batch_size, device=device, max_spec=max_spec
    )
    out = pd.DataFrame(out).sort_values(['Spectra_Similarity']).reset_index()
    print(summ_out)
    print(out)
    if out_csv is not None:
        print(f">>> Saving results to {out_csv}")
        out.to_csv(out_csv, index=False)


def evaluate_on_sptxt(model, filepath, batch_size=4, device="cpu", *args, **kwargs):
    ds = PeptideDataset.from_sptxt(filepath=filepath, *args, **kwargs)
    return evaluate_on_dataset(
        model=model, dataset=ds, batch_size=batch_size, device=device
    )


def evaluate_on_dataset(
    model: PepTransformerModel, dataset: PeptideDataset, batch_size=4, device="cpu"
):
    dl = torch.utils.data.DataLoader(dataset, batch_size)
    cs = torch.nn.CosineSimilarity()

    model.eval()
    model.to(device)
    rt_results = []
    mod_sequences = dataset.df["ModSequences"]
    rt_real = dataset.df["RTs"]
    charges = dataset.df["Charges"]
    spec_results = []

    print(">>> Starting Evaluation of the spectra <<<")
    start_time = time.time()
    with torch.no_grad():
        for b in tqdm(dl):
            outs = model.forward(
                src=b.encoded_sequence.clone().to(device),
                charge=b.charge.clone().to(device),
                mods=b.encoded_mods.clone().to(device),
                nce=b.nce.clone().to(device),
            )

            out_spec = outs.spectra.cpu().clone()
            out_spec = out_spec / out_spec.max(axis=1).values.unsqueeze(0).T

            spec_results.append(cs(out_spec, b.encoded_spectra))
            rt_results.append(outs.irt.cpu().clone())
            del b
            del outs

    end_time = time.time()
    elapsed_time = end_time - start_time

    rt_results = torch.cat(rt_results) * 100
    spec_results = torch.cat(spec_results)

    print(f">> Elapsed time for {len(spec_results)} results was {elapsed_time}.")
    print(
        f">> {len(spec_results) / elapsed_time} results/sec"
        f"; {elapsed_time / len(spec_results)} sec/res"
    )
    out = {
        "ModSequence": mod_sequences,
        "Charges": charges,
        "Predicted_iRT": rt_results.numpy().flatten(),
        "Real_RT": rt_real.to_numpy().flatten(),
        "Spectra_Similarity": spec_results.numpy().flatten(),
    }

    rt_fit = polyfit(norm(out["Predicted_iRT"]), norm(out["Real_RT"]))
    summ_out = {
        "normRT Rsquared": rt_fit["determination"],
        "AverageSpectraSimilarty": out["Spectra_Similarity"].mean(),
    }
    return out, summ_out


def norm(x: ndarray) -> ndarray:
    """Normalizes a numpy array by substracting mean and dividing by standard deviation"""
    return (x - x.mean()) / x.std()


# Polynomial Regression
# Implementation from:
# https://stackoverflow.com/questions/893657/
def polyfit(x, y, degree=1):
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
