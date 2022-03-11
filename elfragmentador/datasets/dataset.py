from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from argparse import _ArgumentGroup
from collections import defaultdict
from os import PathLike
from typing import Iterable, Iterator, List, NamedTuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import uniplot
from pandas.core.frame import DataFrame
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from elfragmentador import encoding_decoding, utils_data
from elfragmentador.datasets.batch_utils import _log_batches
from elfragmentador.metrics import MetricCalculator
from elfragmentador.model import PepTransformerModel
from elfragmentador.named_batches import (
    EvaluationLossBatch,
    EvaluationPredictionBatch,
    ForwardBatch,
    NamedTensorBatch,
    PredictionResults,
    TrainBatch,
)
from elfragmentador.utils_data import cat_collate, terminal_plot_similarity

# TODO write tests so the subclasses actually use nce over-writting ...


class NCEOffsetHolder(ABC):
    @classmethod
    def derived(cls, recursive=True, flatten=False, return_classes=False):
        """
        Lists all derived subclasses as a nested dictionary.

        Returns:
            Dict:

        Examples:
            >>> NCEOffsetHolder.derived(flatten=True)
            ['BatchDataset', 'DatasetBase', 'FastaDataset', 'IterableDatasetBase', \
            'MokapotPSMDataset', ..., 'SpectronautLibrary']
            >>> NCEOffsetHolder.derived(flatten=True, return_classes=True)
            [<class 'elfragmentador.datasets.dataset.BatchDataset'>, ..., \
            <class 'elfragmentador.datasets.spectronaut_dataset.SpectronautLibrary'>]
            >>> NCEOffsetHolder.derived()
            {'NCEOffsetHolder': [{'DatasetBase': ..., {'IterableDatasetBase': \
            [{'PinDataset': [{'MokapotPSMDataset': []}]}]}]}
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

            def key_fun(x):
                return x.__name__ if isinstance(x, type) else x

            outs = sorted(list(set(flat_outs)), key=key_fun)
        else:

            outs = {entry_name: outs}

        return outs

    @property
    def nce_offset(self) -> float:
        if hasattr(self, "_nce_offset"):
            return self._nce_offset
        else:
            return None

    @nce_offset.setter
    def nce_offset(self, value: float):
        msg = f"Setting nce offset to {value}"
        if self.overwrite_nce is not None:
            msg += ", removing nce overwritting"
        logging.info(msg)
        self._nce_offset = value
        self._overwrite_nce = None

    @property
    def overwrite_nce(self) -> float:
        if hasattr(self, "_overwrite_nce"):
            return self._overwrite_nce
        else:
            return None

    @overwrite_nce.setter
    def overwrite_nce(self, value: float):
        msg = f"Setting nce overwritting to {value}"
        if self.nce_offset is not None:
            msg += ", removing nce offset"
        logging.info(msg)
        self._overwrite_nce = value
        self._nce_offset = None

    def calc_nce(self, value: float) -> float:
        self._used_calc_nce = True
        if hasattr(self, "overwrite_nce") and self.overwrite_nce:
            value = self.overwrite_nce
        elif hasattr(self, "nce_offset") and self.nce_offset:
            value = value + self.nce_offset

        return value

    def disable_nce_offset(self) -> None:
        if hasattr(self, "overwrite_nce") and self.overwrite_nce:
            self._backup_nce = {"overwrite_nce": self.overwrite_nce}
            self.overwrite_nce = None
        elif hasattr(self, "nce_offset") and self.nce_offset:
            self._backup_nce = {"nce_offset": self.nce_offset}
            self.nce_offset = None

    def enable_nce_offset(self) -> None:
        if hasattr(self, "_backup_nce"):
            for k, v in self._backup_nce.items():
                setattr(self, k, v)

    @property
    def used_calc_nce(self) -> bool:
        if hasattr(self, "_used_calc_nce") and self._used_calc_nce:
            return True
        else:
            return False

    @abstractmethod
    def greedify(self):
        pass

    @abstractmethod
    def top_n_subset(self, n: int):
        pass

    @abstractmethod
    def append_batches(self, batches: NamedTuple[Tensor], prefix: str = ""):
        pass

    @abstractmethod
    def save_data(self, prefix: PathLike):
        pass

    def optimize_nce(
        self,
        model,
        offsets: Iterable[float] = range(-10, 10, 2),
        n: int = 500,
        predictor: Predictor = None,
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

    def __iter__(self) -> int:
        for i in range(len(self)):
            yield self[i]


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

    def __getitem__(self, index: int) -> Union[ForwardBatch, TrainBatch]:
        out = [x[index] for x in self.batches]
        out = self.batchtype(*out)
        return out

    def __len__(self) -> int:
        return self.length

    def append_batches(self, batches, prefix):
        raise NotImplementedError

    def greedify(self) -> None:
        pass

    def save_data(self, prefix: PathLike):
        raise NotImplementedError

    def top_n_subset(self, n: int):
        raise RuntimeError

    @staticmethod
    def cat_list(tensor_list: List[Tensor]):
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
    def cat_batch_list(batch_list: List[NamedTensorBatch]):
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

    def __init__(
        self, gt_db: Dataset, pred_db: Dataset, ignore_nce=False, *args, **kwargs
    ):
        """
        Generic dataset comparer.

        Provided 2 different datasets, gives utilities to compare them

        It is meant to be used to compare ground-truth vs predicted spectra.
        It calculates the general losses and the scaled retention time loss.

        Args:
            gt_db (NCEOffsetHolder)
            pred_db (NCEOffsetHolder)

        Example:
            >>> # ds1 = PeptideDataset.from_sptxt("path_to_sptxt")
            >>> # ds2 = MokapotPSMDataset("path_to_psms.txt")
            >>> # comp = ComparissonDataset(ds1, ds2)
            >>> # comp.compare()
            >>> # out = comp.save_data()
            >>> # out
            # Contains the columns [Sequence, Charge, NCE, scaled_se_loss, loss_cosine,
            # loss_irt, loss_angle]
        """
        super().__init__()
        self.gt_db = gt_db
        self.pred_db = pred_db

        if len(gt_db) != len(pred_db):
            logging.info(
                "Length of datasets does not match,"
                " attempting to match them by keys (sequence, charge and nce)"
            )
            self._match_datasets(ignore_nce=ignore_nce)
        else:
            self.length = len(pred_db)

    def __getitem__(self, index: int):
        if hasattr(self, "mapping"):
            return self.__match_getitem__(index)
        else:
            return self.__index_getitem__(index)

    def __index_getitem__(self, index: int):
        gt = PredictionResults(
            irt=self.gt_db[index].irt, spectra=self.gt_db[index].spectra
        )
        pred = PredictionResults(
            irt=self.pred_db[index].irt, spectra=self.pred_db[index].spectra
        )
        return {"gt": gt, "pred": pred}

    def __len__(self):
        return self.length

    @staticmethod
    def _make_key(batch: Union[ForwardBatch, TrainBatch], ignore_nce=False):
        mods = F.pad(batch.mods, (0, batch.seq.size(-1) - batch.mods.size(-1)))
        mod_seq = encoding_decoding.decode_mod_seq(
            seq_encoding=batch.seq, mod_encoding=mods
        )
        nce_key = f"/{round(float(batch.nce), 2)}" if not ignore_nce else "/0"
        key = f"{mod_seq}/{int(batch.charge)}{nce_key}"
        return key

    def _match_datasets(self, ignore_nce=False):
        if hasattr(self, "mapping"):
            return None

        base = defaultdict(lambda: {"gt": None, "pred": None, "fw_batch": None})

        self.gt_db.disable_nce_offset()
        self.pred_db.disable_nce_offset()

        gt_keys = []
        for x in tqdm(self.gt_db):
            preds = PredictionResults(irt=x.irt, spectra=x.spectra)
            seqs = ForwardBatch(seq=x.seq, mods=x.mods, charge=x.charge, nce=x.nce)
            key = self._make_key(seqs, ignore_nce=ignore_nce)
            gt_keys.append(key)
            base[key]["gt"] = preds
            base[key]["fw_batch"] = seqs

        pred_keys = []
        for x in tqdm(self.pred_db):
            preds = PredictionResults(irt=x.irt, spectra=x.spectra)
            seqs = ForwardBatch(seq=x.seq, mods=x.mods, charge=x.charge, nce=x.nce)
            key = self._make_key(seqs, ignore_nce=ignore_nce)
            pred_keys.append(key)
            base[key]["pred"] = preds
            base[key]["fw_batch"] = seqs

        pred_keys = set(pred_keys)
        gt_keys = set(gt_keys)

        logging.info(
            (
                f"{len(pred_keys.intersection(gt_keys))} keys in both datasets,",
                f"{len(gt_keys)} in GT, {len(pred_keys)} in Preds",
            )
        )

        logging.info(
            (
                f"{list(pred_keys - gt_keys)[:5]} sample keys not in gt \n",
                f"{list(gt_keys - pred_keys)[:5]} Sample keys not in preds",
            )
        )

        self.mapping = list(base)
        self.mapping_pairs = base

        self.pred_db.enable_nce_offset()
        self.gt_db.enable_nce_offset()

        self.length = len(self.mapping)

    def __match_getitem__(self, index: int):
        mapping_key = self.mapping[index]
        outs = self.mapping_pairs[mapping_key]

        if outs["gt"] is None:
            outs["gt"] = PredictionResults(
                irt=torch.full_like(outs["pred"].irt, float("nan")),
                spectra=torch.full_like(outs["pred"].spectra, float("nan")),
            )

        if outs["pred"] is None:
            outs["pred"] = PredictionResults(
                irt=torch.full_like(outs["gt"].irt, float("nan")),
                spectra=torch.full_like(outs["gt"].spectra, float("nan")),
            )

        gt = PredictionResults(irt=outs["gt"].irt, spectra=outs["gt"].spectra)
        pred = PredictionResults(irt=outs["pred"].irt, spectra=outs["pred"].spectra)
        return {"gt": gt, "pred": pred}

    def compare(
        self, plot: Optional[bool] = False, predictor: Optional[Predictor] = None
    ):
        if predictor is None:
            predictor = Predictor(batch_size=16)

        comp_dl = predictor.make_dataloader(self)
        comparisson_model = MetricCalculator()

        losses = predictor.test(comparisson_model, comp_dl, plot=plot)
        logging.info("Saving calculated losses to {self}.losses")
        self.losses = losses
        return losses

    def save_data(self, prefix: Optional[PathLike] = None):
        """
        Saves the data to a csv file if a path is provided, else it returns the
        data as a dataframe.
        """

        df_dict = {
            "Sequence": [],
            "Charge": [],
            "NCE": [],
        }

        loss_dict = self.losses._asdict()
        [df_dict.update({k: []}) for k, v in loss_dict.items()]

        for i, key in enumerate(self.mapping):

            split_key = key.split("/")

            df_dict["Sequence"].append(split_key[0])
            df_dict["Charge"].append(int(split_key[1]))
            df_dict["NCE"].append(float(split_key[2]))

            for k, v in loss_dict.items():
                df_dict[k].append(float(v[i].squeeze().numpy()))

        out_df = DataFrame(df_dict)
        if prefix is not None:
            logging.info(f"Saving data to {prefix}.csv and {prefix}.feather")
            out_df.to_csv(str(prefix) + ".csv", index=False)
            out_df.reset_index(drop=True).to_feather(str(prefix) + ".feather")
        else:
            logging.info(
                "Not saving data to disk because no prefix"
                " was passed to {self}.save_data"
            )

        return out_df


class Predictor(Trainer):
    """
    A class for testing a model and generating predictions.
    """

    def __init__(
        self,
        gpus: Optional[Union[List[int], str, int]] = 0,
        precision: int = 32,
        batch_size: int = 4,
    ):
        """
        __init__

        Initializes a Predictor object

        Args:
            gpus (Optional[Union[List[int], str, int]]):
                The gpus to use for inference.
            precision (int, optional):
                The precision to use for the model. Defaults to 32.
            batch_size (int, optional):
                The batch size to use for training. Defaults to 4.
        """
        super().__init__(gpus=gpus, precision=precision)
        self.batch_size = batch_size

    @staticmethod
    def add_predictor_args(parser: _ArgumentGroup):
        # Remember to add new arguments to __init__
        parser.add_argument(
            "--gpus",
            default=0,
        )
        parser.add_argument(
            "--precision",
            default=32,
            help=(
                "Precision to use during prediction (32 or 16),"
                " only available using GPU"
            ),
            type=int,
        )
        parser.add_argument(
            "--batch_size",
            default=32,
            help=(
                "Batch size to use during inference"
                " (I suggest ~32 on a cpu and ~600 on a gpu)"
            ),
            type=int,
        )

    def predict_dataset(
        self,
        model: PepTransformerModel,
        dataset: Union[DatasetBase, IterableDatasetBase],
    ) -> PredictionResults:
        """
        predict_dataset.

        Args:
            model (PepTransformerModel): A model to use for prediction
            dataset (Union[DatasetBase, IterableDatasetBase]):
                A dataset to use for prediction

        Returns:
            PredictionResults: A named tuple with the predictions returned by the model
        """
        dl = DataLoader(
            dataset=dataset,
            batch_size=int(self.batch_size),
            collate_fn=utils_data.collate_fun,
        )
        outs = self.predict(model, test_dataloader=dl)

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
        """
        evaluate_dataset.

        Args:
            model (PepTransformerModel):
                A model to use for prediction
            dataset (Union[DatasetBase, IterableDatasetBase]):
                A dataset to use for prediction
            plot (Optional[bool], optional):
                Wether to plot the comparissons. Defaults to True.
            optimize_nce (Optional[Union[bool, Iterable[float]]], optional):
                A range of values to use as offset to optimize the collision energy.
                Defaults to range(-10, 10, 2).
            keep_predictions (Optional[bool], optional):
                Wether to keep the predictions in the dataset,
                requires the implementation of append_batches in the dataset.
                Defaults to False.
            save_prefix (Optional[PathLike], optional):
                A path to use to save the data. Defaults to None.

        Returns:
            Union[EvaluationLossBatch, EvaluationPredictionBatch]
        """
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

    def make_dataloader(self, dataset: DataLoader):
        warnings.filterwarnings(
            "ignore", message=".*The dataloader.*workers.*bottleneck.*"
        )

        logging.info(
            f"Initializig dataloader for {dataset} and batch size of {self.batch_size}"
        )
        dl = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=utils_data.collate_fun,
        )
        return dl

    def predict(
        self, model, test_dataloader: DataLoader, *args, **kwargs
    ) -> PredictionResults:
        outs = super().predict(model, dataloaders=test_dataloader, *args, **kwargs)
        outs = cat_collate(outs)
        outs = PredictionResults(irt=outs.irt * 100, spectra=outs.spectra)

        return outs

    def test(
        self, model, test_dataloader: DataLoader, plot: bool = True, *args, **kwargs
    ) -> EvaluationLossBatch:
        self.plot = plot
        logging.info(">>> Starting Evaluation of the spectra <<<")

        super().test(model, test_dataloader, *args, **kwargs)

        # EvaluationLossBatch, this gets saved by the passed model. when
        # implemented in on_test_epoch_end / on_test_end
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
