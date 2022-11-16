from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Generator, Literal

from ms2ml import AnnotatedPeptideSpectrum
from ms2ml.data.adapters import BaseAdapter

AdapterModes = Literal["predict", "compare"]


class Predictor:
    def predict(
        self, adapter, model, *args, **kwargs
    ) -> Generator[AnnotatedPeptideSpectrum, None, None]:
        pass

    def compare(self, adapter, model, *args, **kwargs) -> Any:
        pass

    def predict_to_file(self, adapter, model, out_filepath, *args, **kwargs) -> None:
        adapter = self.setup_adapter(adapter, *args, **kwargs)

    @singledispatchmethod
    def setup_adapter(
        self, adapter, mode: AdapterModes, *args, **kwargs
    ) -> BaseAdapter:
        pass

    @setup_adapter.register
    def _(self, adapter: str | Path, *args, **kwargs):
        pass

    @setup_adapter.register
    def _(self, adapter: BaseAdapter, *args, **kwargs):
        adapter.out_hook = None
        return adapter

    @staticmethod
    def adapter_out_hook_predict_factory(model):
        def adapter_out_hook_predict(spec):
            pass

        return adapter_out_hook_predict

    @staticmethod
    def adapter_out_hook_compare_factory(model):
        pred_fun = Predictor.adapter_out_hook_predict_factory(model)

        def adapter_out_hook_compare(spec):
            pred_fun(spec)

        return adapter_out_hook_compare
