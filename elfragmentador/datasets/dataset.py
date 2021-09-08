import logging
from os import PathLike
from typing import Optional, Union, List, Iterable
from argparse import _ArgumentGroup
import warnings

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data.dataset import TensorDataset

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

    @abstractmethod
    def greedify(self):
        pass

    @abstractmethod
    def top_n_subset(self, n):
        pass

    @abstractmethod
    def append_batches(self, batches):
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
        elem = batches[0]
        for x in elem:
            assert isinstance(x, Tensor)

        lens = list(set([len(x) for x in elem]))
        assert len(lens) == 1

        keep_batches = [
            self.cat_list([y[i] for y in batches]) for i, _ in enumerate(elem)
        ]

        self.length = len(keep_batches[0])
        self.batchtype = type(elem)
        self.batches = self.batchtype(*keep_batches)
        info = {k: v.shape for k, v in self.batches._asdict().items()}
        logging.info(f"Initialized BatchDataset with {info}")

    def __getitem__(self, index) -> Union[ForwardBatch, TrainBatch]:
        out = [x[index] for x in self.batches]
        out = self.batchtype(*out)
        return out

    def __len__(self):
        return self.length

    def append_batches(self, batches):
        raise RuntimeError

    def greedify(self):
        pass

    def save_data(self, prefix: PathLike):
        raise RuntimeError

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
        keep_predictions: Optional[bool] = False,
        save_prefix: Optional[PathLike] = None,
    ) -> Union[EvaluationLossBatch, EvaluationPredictionBatch]:
        self.plot = plot
        if optimize_nce:
            best_nce_offset = dataset.optimize_nce(model, optimize_nce, predictor=self)
            dataset.nce_offset = best_nce_offset

        dl = self.make_dataloader(dataset)

        if keep_predictions:
            tmp_ds = BatchDataset([x for x in dl])
            preds = self.predict(model, test_dataloader=self.make_dataloader(tmp_ds))
            {
                logging.info(f"Predictions: {k}:{v.shape}")
                for k, v in preds._asdict().items()
            }

            comp_ds = ComparissonDataset(tmp_ds, BatchDataset([preds]))
            comparisson_model = MetricCalculator()
            outs = self.test(
                comparisson_model, self.make_dataloader(comp_ds), plot=plot
            )
            {
                logging.info(f"Comparissons: {k}:{v.shape}")
                for k, v in outs._asdict().items()
            }
            outs = EvaluationPredictionBatch(**preds._asdict(), **outs._asdict())
        else:
            outs = self.test(model, dl, plot=plot)

        if save_prefix is not None:
            dataset.append_batches(outs)
            dataset.save_data(save_prefix)

        return outs

    def make_dataloader(self, dataset):
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
