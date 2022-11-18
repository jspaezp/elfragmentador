from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Generator, Iterable, Literal

import pandas as pd
import torch
import uniplot
from ms2ml import AnnotatedPeptideSpectrum
from ms2ml.data.adapters import BaseAdapter, read_data
from ms2ml.data.parsing.encyclopedia import write_encyclopedia
from ms2ml.metrics.base import spectral_angle
from ms2ml.utils import allign_intensities
from tqdm.auto import tqdm

from elfragmentador.config import CONFIG
from elfragmentador.data.converter import DeTensorizer, Tensorizer
from elfragmentador.model import PepTransformerBase, PepTransformerModel
from elfragmentador.version import __version__

AdapterModes = Literal["predict", "compare"]


class Predictor:
    def __init__(self, model: PepTransformerModel | PepTransformerBase) -> None:
        """Utility class to predict spectra and compare predictions

        Examples:
            >>> model = PepTransformerModel(d_model=96)
            >>> predictor = Predictor(model=model)
            >>> # predictor.predict_to_file("foo.fasta", "out.dlib", nce=30)
            >>> # predictor.compare("out.dlib")

        """
        self.model = model

    def predict(
        self, adapter, nce, config=CONFIG, *args, **kwargs
    ) -> Generator[AnnotatedPeptideSpectrum, None, None]:
        adapter_iter = self.setup_adapter(
            adapter,
            model=self.model,
            mode="predict",
            nce=nce,
            config=config,
            *args,
            **kwargs,
        )
        yield from adapter_iter

    def screen_nce(self, adapter, nce_list, *args, **kwargs) -> pd.DataFrame:
        best_nce = None
        best_sa = 0
        for nce in nce_list:
            outs = (
                self.compare(adapter=adapter, nce=nce, max_spec=1000, *args, **kwargs),
            )
            med_sa = outs["spectral angle"].median()
            if med_sa > best_sa:
                best_sa = med_sa
                best_nce = nce

        return best_nce

    def compare(
        self, adapter, nce, config=CONFIG, max_spec=float("inf"), *args, **kwargs
    ) -> pd.DataFrame:
        if isinstance(nce, Iterable):
            if len(nce) > 1:
                nce = self.screen_nce(adapter, nce, *args, **kwargs)
            else:
                nce = nce[0]

        adapter_iter = self.setup_adapter(
            adapter,
            model=self.model,
            mode="compare",
            nce=nce,
            config=config,
            *args,
            **kwargs,
        )
        out = []
        for i, x in enumerate(adapter_iter):
            out.append(x)
            if i > max_spec:
                break
        df = pd.DataFrame(out)

        uniplot.plot(ys=df["pred rt"], xs=df["rt"], title="Pred RT (y) vs RT (x)")

        uniplot.histogram(
            df["spectral angle"], title="Histogram of the spectral angles"
        )
        return df

    def compare_to_file(self, adapter, out_filepath, nce, *args, **kwargs) -> None:
        if not str(out_filepath).endswith(".csv"):
            raise ValueError(
                "The output file for the comparison must be 'csv' to denote it as a"
                "comma-separated values file"
            )
        df = self.compare(adapter=adapter, nce=nce, *args, **kwargs)
        df.to_csv(out_filepath)

    def predict_to_file(
        self, adapter, out_filepath: Path | str, nce: float, *args, **kwargs
    ) -> None:
        if not str(out_filepath).endswith("dlib"):
            raise ValueError(
                "The output file for the prediction must be 'dlib' to denote it as a"
                "EncyclopeDIA File DDA library"
            )
        source_filename = f"ElFragmentador.v{__version__}"
        write_encyclopedia(
            spectra=self.predict(adapter=adapter, nce=nce, *args, **kwargs),
            file=out_filepath,
            source_file=source_filename,
        )

    def setup_adapter(
        self,
        adapter,
        mode: AdapterModes,
        model,
        nce,
        charges=None,
        config=CONFIG,
        *args,
        **kwargs,
    ) -> BaseAdapter:
        if isinstance(adapter, Path):
            adapter = str(adapter)
        if isinstance(adapter, str):
            config = dataclasses.replace(config)
            if charges:
                config.precursor_charges = charges
            adapter = read_data(adapter, config=config, *args, **kwargs)

        for i, _ in enumerate(adapter.parse()):
            continue

        length = i + 1

        if mode == "predict":
            adapter.out_hook = self.adapter_out_hook_predict_factory(
                model=model, nce=nce
            )
        elif mode == "compare":
            adapter.out_hook = self.adapter_out_hook_compare_factory(
                model=model, nce=nce
            )
        return tqdm(adapter.parse(), total=length)

    @staticmethod
    def adapter_out_hook_predict_factory(model, nce):
        tmp_tensorizer = Tensorizer()
        tmp_detensorizer = DeTensorizer()

        @torch.no_grad()
        def adapter_out_hook_predict(
            spec: AnnotatedPeptideSpectrum,
        ) -> AnnotatedPeptideSpectrum:
            tensor_batch = tmp_tensorizer(spec, nce=nce)
            pred = model.forward(
                seq=tensor_batch.seq,
                mods=tensor_batch.mods,
                nce=tensor_batch.nce,
                charge=tensor_batch.charge,
            )
            out_spec = tmp_detensorizer.make_spectrum(
                seq=tensor_batch.seq,
                mod=tensor_batch.mods,
                charge=tensor_batch.charge,
                fragment_vector=pred.spectra,
                irt=pred.irt,
            )
            # rt prediciton from the model is BIOGNOSYS IRT/100, which is in "mins"
            # so *100 makes it "minutes", *60 makes it seconds
            return out_spec

        return adapter_out_hook_predict

    @staticmethod
    def adapter_out_hook_compare_factory(model, nce):
        pred_fun = Predictor.adapter_out_hook_predict_factory(model, nce=nce)

        def adapter_out_hook_compare(spec: AnnotatedPeptideSpectrum):
            # predict
            pred_spec = pred_fun(spec)
            # compare predictions ...
            ints = allign_intensities(
                mz1=spec.mz,
                mz2=pred_spec.mz,
                int1=spec.intensity,
                int2=pred_spec.intensity,
                tolerance=CONFIG.g_tolerances[1],
                unit=CONFIG.g_tolerance_units[1],
            )
            sa_metric = spectral_angle(*ints)
            rt = pred_spec.retention_time
            return {
                "peptide sequence": spec.precursor_peptide.to_proforma(),
                "spectral angle": sa_metric,
                "pred rt": rt.seconds(),
                "rt": spec.retention_time.seconds(),
            }

        return adapter_out_hook_compare
