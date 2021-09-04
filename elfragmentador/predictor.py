import time
import logging
from elfragmentador.named_batches import EvaluationLossBatch, PredictionResults
from elfragmentador.model import PepTransformerModel
from elfragmentador.datasets import IterableDatasetBase, DatasetBase
from elfragmentador import utils_data
from typing import Optional, Union, List
from argparse import ArgumentParser, _ArgumentGroup
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from elfragmentador.utils_data import cat_collate, terminal_plot_similarity
import uniplot


class Predictor(Trainer):
    def __init__(
        self, gpus: Optional[Union[List[int], str, int]] = 0, precision: int = 32
    ):
        super().__init__(gpus=gpus, precision=precision)

    @staticmethod
    def add_predictor_args(parser: _ArgumentGroup):
        parser.add_argument("--gpus", default=0)
        parser.add_argument(
            "--precision",
            default=32,
            help="Precision to use during prediction (32 or 16), only available using GPU",
        )

    def predict_dataset(
        self,
        model: PepTransformerModel,
        dataset: Union[DatasetBase, IterableDatasetBase],
        batch_size: int,
    ) -> PredictionResults:
        dl = DataLoader(
            dataset=dataset, batch_size=batch_size, collate_fn=utils_data.collate_fun
        )
        outs = self.predict(model, dl)

        return outs

    def evaluate_dataset(
        self,
        model: PepTransformerModel,
        dataset: Union[DatasetBase, IterableDatasetBase],
        batch_size: int,
        plot: Optional[bool] = True,
    ) -> EvaluationLossBatch:
        dl = DataLoader(
            dataset=dataset, batch_size=batch_size, collate_fn=utils_data.collate_fun
        )
        outs = self.test(model, dl, plot=plot)

        return outs

    def predict(self, model, test_dataloader, *args, **kwargs) -> PredictionResults:
        outs = super().predict(model, test_dataloader, *args, **kwargs)
        outs = cat_collate(outs)
        outs = PredictionResults(irt=outs.irt * 100, spectra=outs.spectra)

        return outs

    def test(
        self, model, test_dataloader, plot=True, *args, **kwargs
    ) -> EvaluationLossBatch:
        self.plot = plot
        logging.info(">>> Starting Evaluation of the spectra <<<")

        super().test(model, test_dataloader, *args, **kwargs)

        test_results = self.test_results

        # ["scaled_se_loss", "loss_cosine", "loss_irt", "loss_angle"]
        self.test_results = None

        if plot:
            nonmissing_se_loss = (
                test_results.scaled_se_loss[~torch.isnan(test_results.scaled_se_loss)]
                .cpu()
                .numpy()
            )
            terminal_plot_similarity(nonmissing_se_loss, "Square scaled RT error")
            terminal_plot_similarity(
                1 - test_results.loss_cosine.cpu().detach().numpy(),
                "Spectra Cosine Similarity",
            )
            terminal_plot_similarity(
                1 - test_results.loss_angle.cpu().detach().numpy(), "Spectral Angle"
            )

        return test_results
