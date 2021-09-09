import logging
from os import PathLike
from typing import Optional, Union, List, Iterable, Iterator
from argparse import _ArgumentGroup
import warnings

from abc import ABC, abstractmethod

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from elfragmentador.utils_data import cat_collate, terminal_plot_similarity
from elfragmentador.metrics import MetricCalculator
import uniplot

from elfragmentador import spectra, utils_data
from elfragmentador.model import PepTransformerModel
from elfragmentador.named_batches import (
    EvaluationLossBatch,
    EvaluationPredictionBatch,
    ForwardBatch,
    PredictionResults,
    TrainBatch,
)


# TODO write tests so the subclasses actually use nce over-writting ...


class NCEOffsetHolder(ABC):
    @classmethod
    def derived(cls, recursive=True, flatten=False, return_classes=False):
        """
        Lists all derived subclasses as a nested dictionary

        Returns:
            Dict:

        Examples:
            >>> NCEOffsetHolder.derived(flatten=True)
            ['BatchDataset', 'DatasetBase', 'IterableDatasetBase', 'MokapotPSMDataset', 'NCEOffsetHolder', 'PeptideDataset', 'PinDataset', 'SequenceDataset']
            >>> NCEOffsetHolder.derived(flatten=True, return_classes=True)
            [<class 'elfragmentador.datasets.dataset.BatchDataset'>, ..., <class 'elfragmentador.datasets.sequence_dataset.SequenceDataset'>]
            >>> NCEOffsetHolder.derived()
            {'NCEOffsetHolder': [{'DatasetBase': [{'BatchDataset': []}, {'PeptideDataset': []}, {'SequenceDataset': []}]}, {'IterableDatasetBase': [{'PinDataset': [{'MokapotPSMDataset': []}]}]}]}
        """

        entry_name = cls if return_classes else cls.__name__

        outs = list(cls.__subclasses__())
        outs = [
            subclass.derived(
                recursive=recursive, flatten=flatten, return_classes=return_classes
            )
            if recursive
            else subclass
            for subclass in outs
        ]

        if not return_classes:
            outs = [x.__name__ if hasattr(x, "__name__") else x for x in outs]

        if flatten:
            flat_outs = []
            flat_outs.extend([entry_name])

            for out in outs:
                if isinstance(out, list):
                    flat_outs.extend(out)
                else:
                    flat_outs.extend([out])

            if any([isinstance(x, list) for x in flat_outs]):
                breakpoint()

            key = lambda x: x.__name__ if isinstance(x, type) else x
            outs = sorted(list(set(flat_outs)), key=key)
        else:

            outs = {entry_name: outs}

        return outs

    @property
    def nce_offset(self):
        if hasattr(self, "_nce_offset"):
            return self._nce_offset
        else:
            return None

    @nce_offset.setter
    def nce_offset(self, value):
        msg = f"Setting nce offset to {value}"
        if self.overwrite_nce is not None:
            msg += ", removing nce overwritting"
        logging.info(msg)
        self._nce_offset = value
        self._overwrite_nce = None

    @property
    def overwrite_nce(self):
        if hasattr(self, "_overwrite_nce"):
            return self._overwrite_nce
        else:
            return None

    @overwrite_nce.setter
    def overwrite_nce(self, value):
        msg = f"Setting nce overwritting to {value}"
        if self.nce_offset is not None:
            msg += ", removing nce offset"
        logging.info(msg)
        self._overwrite_nce = value
        self._nce_offset = None

    def calc_nce(self, value):
        self._used_calc_nce = True
        if hasattr(self, "overwrite_nce") and self.overwrite_nce:
            value = self.overwrite_nce
        elif hasattr(self, "nce_offset") and self.nce_offset:
            value = value + self.nce_offset

        return value

    @property
    def used_calc_nce(self):
        if hasattr(self, "_used_calc_nce") and self._used_calc_nce:
            return True
        else:
            return False

    @abstractmethod
    def greedify(self):
        pass

    @abstractmethod
    def top_n_subset(self, n):
        pass

    @abstractmethod
    def append_batches(self, batches, prefix=""):
        pass

    @abstractmethod
    def save_data(self, prefix: PathLike):
        pass

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

            if not tmp_ds.used_calc_nce:
                logging.error(f"calc_nce was not used by {tmp_ds}")

        msg = f"Best nce offset was {best}, score = {best_score}"
        logging.info(msg)

        if len(tried_offsets) > 1:
            uniplot.plot(ys=score_history, xs=tried_offsets)

        self.nce_offset = best
        return best


# TODO split inference and prediction datasets
class DatasetBase(Dataset, NCEOffsetHolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __getitem__(self, index) -> Union[ForwardBatch, TrainBatch]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class IterableDatasetBase(IterableDataset, NCEOffsetHolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __iter__(self) -> Iterator[ForwardBatch]:
        raise NotImplementedError


class BatchDataset(DatasetBase):
    def __init__(self, batches: List[ForwardBatch], *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info("Initializing BatchDataset")

        keep_batches = self.cat_batch_list(batches)

        self.length = len(keep_batches[0])
        self.batchtype = type(keep_batches)
        self.batches = keep_batches
        info = {k: v.shape for k, v in self.batches._asdict().items()}
        logging.debug(f"Initialized BatchDataset with {info}")

    def __getitem__(self, index) -> Union[ForwardBatch, TrainBatch]:
        out = [x[index] for x in self.batches]
        out = self.batchtype(*out)
        return out

    def __len__(self):
        return self.length

    def append_batches(self, batches, prefix):
        raise NotImplementedError

    def greedify(self):
        pass

    def save_data(self, prefix: PathLike):
        raise NotImplementedError

    def top_n_subset(self, n):
        raise RuntimeError

    @staticmethod
    def cat_list(tensor_list):
        lengths = np.array([x.shape for x in tensor_list])
        max_lenghts = lengths.max(axis=0)
        out = []
        for x in tensor_list:
            delta_shape = max_lenghts[1:] - x.shape[1:]
            padding = [[0, 0]] + [(0, d) for d in delta_shape]
            padding = tuple(padding)
            out.append(np.pad(x.numpy(), padding, "constant"))

        out = torch.cat([torch.from_numpy(x) for x in out], dim=0).squeeze()
        if len(out.shape) == 1:
            out = out.unsqueeze(-1)
        return out

    @staticmethod
    def cat_batch_list(batch_list):
        elem = batch_list[0]
        batchtype = type(elem)
        for x in elem:
            assert isinstance(x, Tensor)

        lens = list(set([len(x) for x in elem]))
        assert len(lens) == 1

        keep_batches = [
            BatchDataset.cat_list([y[i] for y in batch_list])
            for i, _ in enumerate(elem)
        ]

        keep_batches = batchtype(*keep_batches)
        return keep_batches


class ComparissonDataset(Dataset):
    accepted_fields = {"spec", "irt"}

    def __init__(self, gt_db, pred_db, *args, **kwargs):
        super().__init__()
        self.gt_db = gt_db
        self.pred_db = pred_db
        assert len(gt_db) == len(pred_db)

    def __getitem__(self, index):
        gt = PredictionResults(
            irt=self.gt_db[index].irt, spectra=self.gt_db[index].spectra
        )
        pred = PredictionResults(
            irt=self.pred_db[index].irt, spectra=self.pred_db[index].spectra
        )
        return {"gt": gt, "pred": pred}

    def __len__(self):
        return len(self.pred_db)


def _log_batches(batches, prefix="Tensor"):
    {logging.debug(f"{prefix}: {k}:{v.shape}") for k, v in batches._asdict().items()}


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
        # Remember to add new arguments to __init__
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
        keep_predictions: Optional[bool] = False,
        save_prefix: Optional[PathLike] = None,
    ) -> Union[EvaluationLossBatch, EvaluationPredictionBatch]:
        self.plot = plot
        if optimize_nce:
            best_nce_offset = dataset.optimize_nce(model, optimize_nce, predictor=self)
            dataset.nce_offset = best_nce_offset

        dl = self.make_dataloader(dataset)

        if keep_predictions:
            gt_batches = BatchDataset.cat_batch_list([x for x in dl])
            dataset.append_batches(gt_batches, prefix="GroundTruth_")
            gt_ds = BatchDataset([gt_batches])

            preds = self.predict(model, test_dataloader=self.make_dataloader(gt_ds))
            dataset.append_batches(preds, prefix="Prediction_")
            _log_batches(preds, "Predictions")

            comp_ds = ComparissonDataset(gt_ds, BatchDataset([preds]))
            comp_dl = self.make_dataloader(comp_ds)
            comparisson_model = MetricCalculator()

            losses = self.test(comparisson_model, comp_dl, plot=plot)
            dataset.append_batches(losses, prefix="Loss_")
            _log_batches(losses, "Losses")

            outs = EvaluationPredictionBatch(**preds._asdict(), **losses._asdict())
        else:
            outs = self.test(model, dl, plot=plot)
            _log_batches(outs, "Losses")

        _log_batches(outs, "Outputs")

        if save_prefix is not None:
            logging.debug("Saving data with prefix '{prefix}'")
            dataset.save_data(save_prefix)

        return outs

    def make_dataloader(self, dataset):
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )

        dl = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=utils_data.collate_fun,
        )
        return dl

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

        # EvaluationLossBatch
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
                1 - test_results.loss_angle.cpu().detach().numpy(),
                "Spectral Angle Similarity",
            )

        return test_results
