from pathlib import Path
import logging
import logging.config

import argparse
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)

import elfragmentador

try:
    from argparse import BooleanOptionalAction
except ImportError:
    # Exception for py <3.8 compatibility ...
    BooleanOptionalAction = "store_true"


import warnings

import pytorch_lightning as pl
import pandas as pd

from elfragmentador.train import build_train_parser, main_train
from elfragmentador.model import PepTransformerModel
from elfragmentador.spectra import sptxt_to_csv
from elfragmentador.utils import append_preds, predict_df
from elfragmentador import datamodules, evaluate, rt

import uniplot

DEFAULT_LOGGER_BASIC_CONF = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": logging.INFO,
}


def _common_checkpoint_args(parser):
    """Adds the common to handle model checkpoints to a parser"""
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=elfragmentador.DEFAULT_CHECKPOINT,
        help="Model checkpoint to use for the prediction, if nothing is passed will download a pretrained model",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to move the model to during the evaluation",
    )


def greeting():
    logging.info(f"ElFragmentador version: {elfragmentador.__version__}")


def _calculate_irt_parser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        prog="elfragmentador_calculate_irt",
    )
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        nargs="+",
        help="Input file(s) to convert (skyline csv output)",
    )
    parser.add_argument(
        "--out",
        default="out.csv",
        type=str,
        help="Name of the file where the output will be written (csv)",
    )
    return parser


def _gen_cli_help(parser):
    def decorate(fun):
        fun.__doc__ = "```\n" + parser.format_help() + "\n```"
        return fun

    return decorate


@_gen_cli_help(_calculate_irt_parser())
def calculate_irt():
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()
    parser = _calculate_irt_parser()

    args = parser.parse_args()
    files = [x.name for x in args.file]
    df = rt.calculate_multifile_iRT(files)
    df.to_csv(str(args.out))


def _append_prediction_parser():
    parser = ArgumentParser(
        prog="elfragmentador_append_pin", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pin",
        type=str,
        help="Input percolator file",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Input percolator file",
    )
    _common_checkpoint_args(parser)
    return parser


@_gen_cli_help(_append_prediction_parser())
def append_predictions():
    """Appends the cosine similarity between the predicted and actual spectra
    to a percolator input.
    """
    log_conf = DEFAULT_LOGGER_BASIC_CONF.copy()
    log_conf.update({"level": logging.INFO})
    logging.basicConfig(**log_conf)
    greeting()

    parser = _append_prediction_parser()
    args = parser.parse_args()

    model = PepTransformerModel.load_from_checkpoint(args.model_checkpoint)
    model.eval()

    out_df = append_preds(in_pin=args.pin, out_pin=args.out, model=model)
    logging.info(out_df)


def _predict_csv_parser():
    parser = ArgumentParser(
        prog="elfragmentador_predict_csv", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Input csv file",
    )
    parser.add_argument(
        "--impute_collision_energy",
        type=float,
        default=0,
        help="Collision energy to use if none is specified in the file",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output .sptxt file",
    )
    _common_checkpoint_args(parser)
    return parser


@_gen_cli_help(_predict_csv_parser())
def predict_csv():
    """Predicts the peptides in a csv file"""
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()

    parser = _predict_csv_parser()
    args = parser.parse_args()
    if args.impute_collision_energy == 0:
        nce = False
    else:
        nce = args.impute_collision_energy

    model = PepTransformerModel.load_from_checkpoint(args.model_checkpoint)
    model.eval()

    with open(args.out, "w") as f:
        f.write(
            predict_df(pd.read_csv(args.csv), impute_collision_energy=nce, model=model)
        )


def _convert_sptxt_parser():
    parser = ArgumentParser(
        prog="elfragmentador_convert_sptxt",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        nargs="+",
        help="Input file(s) to convert (sptxt)",
    )
    parser.add_argument(
        "--warn",
        default=False,
        action=BooleanOptionalAction,
        help="Wether to show warnings or not",
    )
    parser.add_argument(
        "--keep_irts",
        default=False,
        action=BooleanOptionalAction,
        help="Wether to remove sequences that match procal and biognosys iRT peptides",
    )
    parser.add_argument(
        "--min_peaks",
        default=3,
        type=int,
        help="Minimum number of annotated peaks required to keep the spectrum",
    )
    parser.add_argument(
        "--min_delta_ascore",
        default=20,
        type=int,
        help="Minimum ascore required to keep a spectrum",
    )
    return parser


@_gen_cli_help(_convert_sptxt_parser())
def convert_sptxt():
    """convert_sptxt Provides a CLI to convert an sptxt to a csv for training

    provides a CLI for the sptxt_to_csv function, chek that guy out for the actual
    implementation
    """
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()

    parser = _convert_sptxt_parser()
    args = parser.parse_args()

    logging.info([x.name for x in args.file])

    # Here we make the partial function that will be used to actually read the data
    converter = lambda fname, outname: sptxt_to_csv(
        fname,
        outname,
        filter_irt_peptides=args.keep_irts,
        min_delta_ascore=args.min_delta_ascore,
        min_peaks=args.min_peaks,
    )

    for f in args.file:
        out_file = f.name + ".csv"
        if Path(out_file).exists():
            logging.info(
                f"Skipping conversion of '{f.name}' "
                f"to '{out_file}', "
                f"because {out_file} exists."
            )
        else:
            logging.info(f"Converting '{f.name}' to '{out_file}'")
            if args.warn:
                logging.warning("Warning stuff ...")
                converter(f.name, out_file)
            else:
                with warnings.catch_warnings(record=True) as c:
                    logging.warning("Not Warning stuff ...")
                    converter(f.name, out_file)

                if len(c) > 0:
                    logging.warning(
                        f"Last Error Message of {len(c)}: '{c[-1].message}'"
                    )


def _evaluate_parser():
    parser = evaluate.build_evaluate_parser()
    _common_checkpoint_args(parser)
    return parser


@_gen_cli_help(_evaluate_parser())
def evaluate_checkpoint():
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    pl.seed_everything(2020)
    greeting()

    parser = _evaluate_parser()
    args = parser.parse_args()
    dict_args = vars(args)
    logging.info(dict_args)

    model = PepTransformerModel.load_from_checkpoint(args.checkpoint_path)
    if dict_args["csv"] is not None:
        ds = datamodules.PeptideDataset.from_csv(
            args.csv,
            max_spec=args.max_spec,
        )
    elif dict_args["sptxt"] is not None:
        ds = datamodules.PeptideDataset.from_sptxt(
            args.sptxt,
            max_spec=args.max_spec,
        )
    else:
        raise ValueError("Must have an argument to either --csv or --sptxt")

    if dict_args["screen_nce"] is not None:
        nces = [float(x) for x in dict_args["screen_nce"].split(",")]
    elif dict_args["overwrite_nce"] is not None:
        nces = [dict_args["overwrite_nce"]]
    else:
        nces = [False]

    best_res = tuple([{}, {"AverageSpectraCosineSimilarity": 0}])
    best_nce = None
    res_history = []
    for nce in nces:
        if nce:
            logging.info(f">>>> Starting evaluation of NCE={nce}")
        res = evaluate.evaluate_on_dataset(
            model=model,
            dataset=ds,
            batch_size=args.batch_size,
            device=args.device,
            overwrite_nce=nce,
        )
        res_history.append(res[1]["AverageSpectraCosineSimilarity"])
        if (
            res[1]["AverageSpectraCosineSimilarity"]
            > best_res[1]["AverageSpectraCosineSimilarity"]
        ):
            best_res = res
            best_nce = nce

    if len(nces) > 1:
        logging.info(f"Best Nce was {best_nce}")
        uniplot.plot(ys=res_history, xs=nces)

    if dict_args["out_csv"] is not None:
        best_res[0].to_csv(dict_args["out_csv"], index=False)


@_gen_cli_help(build_train_parser())
def train():
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()

    pl.seed_everything(2020)
    parser = build_train_parser()
    args = parser.parse_args()
    dict_args = vars(args)
    logging.info("====== Passed command line args/params =====")
    for k, v in dict_args.items():
        logging.info(f">> {k}: {v}")

    mod = PepTransformerModel(**dict_args)
    if args.from_checkpoint is not None:
        logging.info(f">> Resuming training from checkpoint {args.from_checkpoint} <<")
        weights_mod = PepTransformerModel.load_from_checkpoint(args.from_checkpoint)
        mod.load_state_dict(weights_mod.state_dict())
        del weights_mod

    main_train(mod, args)
