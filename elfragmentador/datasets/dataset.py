import logging
from typing import Optional, Union, List, Iterable
from argparse import _ArgumentGroup


import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import Trainer

from elfragmentador.utils_data import cat_collate, terminal_plot_similarity
import uniplot

from elfragmentador import utils_data
from elfragmentador.model import PepTransformerModel
from elfragmentador.named_batches import EvaluationLossBatch, PredictionResults


# TODO write tests so the subclasses actually use nce over-writting ...


class NCEOffsetHolder:
    @property
    def nce_offset(self):
        return self._nce_offset

    @nce_offset.setter
    def nce_offset(self, value):
        logging.warning(f"Setting nce offset to {value}, removing nce overwritting")
        self._nce_offset = value
        self._overwrite_nce = None

    @property
    def overwrite_nce(self):
        return self._overwrite_nce

    @overwrite_nce.setter
    def overwrite_nce(self, value):
        logging.warning(f"Setting nce overwritting to {value}, removing nce offset")
        self._overwrite_nce = value
        self._nce_offset = None

    def calc_nce(self, value):
        if hasattr(self, "overwrite_nce") and self.overwrite_nce:
            value = self.overwrite_nce
        elif hasattr(self, "nce_offset") and self.nce_offset:
            value = value + self.nce_offset

        return value

    def greedify(self, *args, **kwargs):
        raise NotImplementedError

    def top_n_subset(self, n, metric):
        raise NotImplementedError

    def optimize_nce(
        self,
        model,
        offsets=range(-10, 10, 2),
        n=500,
        predictor=None,
    ):
        # Requires implementing a 'top_n_subset' method in the
        # derived class, optionally have a .metric attribute
        # and a .greedify() if the class has a lazy and a greedy mode
        if predictor is None:
            predictor = Predictor()

        offsets = [0] + list(offsets)
        logging.info(f"Finding best nce offset from {offsets} in {type(self)}")
        best = 0
        best_score = 0

        tmp_ds = self.top_n_subset(n=min(len(self), n))

        try:
            tmp_ds.greedify()
        except NotImplementedError as e:
            logging.warning(
                f"No implemented greedyfing for dataset of class {type(self)}: {e}"
            )

        tried_offsets = []
        score_history = []
        for i, offset in enumerate(offsets):
            if offset in tried_offsets:
                continue

            tried_offsets.append(offset)
            # TODO write a test here
            tmp_ds.nce_offset = offset
            outs = predictor.evaluate_dataset(
                model, tmp_ds, plot=False, optimize_nce=False
            )
            score = 1 - outs.loss_angle.median()
            score_history.append(score)
            logging.info(f"NCE Offset={offset}, score={score}")
            if score > best_score:
                if i > 0:
                    logging.info(
                        (
                            f"Updating best offset (from {best} to {offset}) "
                            f"because {best_score} < {score}"
                        )
                    )
                best = offset
                best_score = score

        msg = f"Best nce offset was {best}, score = {best_score}"
        logging.info(msg)

        if len(tried_offsets) > 1:
            uniplot.plot(ys=score_history, xs=tried_offsets)

        self.nce_offset = best
        return best


class DatasetBase(Dataset, NCEOffsetHolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDatasetBase(IterableDataset, NCEOffsetHolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __iter__(self):
        raise NotImplementedError


class Predictor(Trainer):
    def __init__(
        self,
        gpus: Optional[Union[List[int], str, int]] = 0,
        precision: int = 32,
        batch_size: int = 4,
    ):
        super().__init__(gpus=gpus, precision=precision)
        self.batch_size = batch_size

    @staticmethod
    def add_predictor_args(parser: _ArgumentGroup):
        parser.add_argument("--gpus", default=0)
        parser.add_argument(
            "--precision",
            default=32,
            help="Precision to use during prediction (32 or 16), only available using GPU",
        )
        parser.add_argument(
            "--batch_size",
            default=32,
            help="Precision to use during prediction (32 or 16), only available using GPU",
        )

    def predict_dataset(
        self,
        model: PepTransformerModel,
        dataset: Union[DatasetBase, IterableDatasetBase],
    ) -> PredictionResults:
        dl = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=utils_data.collate_fun,
        )
        outs = self.predict(model, dl)

        return outs

    def evaluate_dataset(
        self,
        model: PepTransformerModel,
        dataset: Union[DatasetBase, IterableDatasetBase],
        plot: Optional[bool] = True,
        optimize_nce: Optional[Union[bool, Iterable[float]]] = range(-10, 10, 2),
    ) -> EvaluationLossBatch:
        if optimize_nce:
            best_nce_offset = dataset.optimize_nce(model, optimize_nce, predictor=self)
            dataset.nce_offset = best_nce_offset

        dl = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=utils_data.collate_fun,
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
