import torch
import numpy as np
from elfragmentador.model import PepTransformerModel
from elfragmentador.datamodules import PeptideDataset

# Given a model checkpoint and some input data, parse the data and return metrics, also a csv with the report


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
    rt_real = dataset.df["RTs"]
    spec_results = []
    with torch.no_grad():
        for b in dl:
            outs = model.forward(
                src=b.encoded_sequence.clone().to(device),
                charge=b.charge.clone().to(device),
                mods=b.encoded_mods.clone().to(device),
                nce=b.nce.clone().to(device),
            )

            spec_results.append(cs(outs.spectra.cpu().clone(), b.encoded_spectra))
            rt_results.append(outs.irt.cpu().clone())
            del b
            del outs

    rt_results = torch.cat(rt_results) * 100
    spec_results = torch.cat(spec_results)
    out = {
        "PredictedRT": rt_results.numpy().flatten(),
        "RealRT": rt_real.to_numpy().flatten(),
        "SpectraSimilarity": spec_results.numpy().flatten(),
    }

    summ_out = {
        "normRT Rsquared": polyfit(norm(out["PredictedRT"]), norm(out["RealRT"]))[
            "determination"
        ],
        "AverageSpectraSimilarty": out["SpectraSimilarity"].mean(),
    }
    return out, summ_out


def norm(x):
    return (x - x.mean()) / x.std()


# Polynomial Regression
# Implementation from:
# https://stackoverflow.com/questions/893657/
def polyfit(x, y, degree=1):
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
