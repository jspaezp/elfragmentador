import torch
from elfragmentador.model import PepTransformerModel
from elfragmentador import datamodules

# Given a model checkpoint and some input data, parse the data and return metrics, also a csv with the report


def evaluate_on_sptxt(model: PepTransformerModel, dataset, batch_size=4, device="cpu"):
    dl = torch.utils.data.DataLoader(dataset, batch_size)
    cs = torch.nn.CosineSimilarity()

    model.eval()
    model.to(device)
    rt_results = []
    spec_results = []
    with torch.no_grad():
        for b in dl:
            outs = model.forward(
                src=b.encoded_seq.clone().to(device),
                charge=b.charge.clone().to(device),
                mods=b.encoded_modm.clone().to(device),
                nce=b.nce.clone().to(device),
            )

            spec_results.append(cs(outs.spectra.cpu().clone(), b.encoded_spectra))
            rt_results.append(outs.irt.cpu().clone())
            del b
            del outs

    rt_results = torch.cat(rt_results)
    spec_results = torch.cat(spec_results)
    out = {
        "PredictedRT": rt_results.numpy(),
        "SpectraSimilarity": spec_results.numpy(),
    }
    return out
