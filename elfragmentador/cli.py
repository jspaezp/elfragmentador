import argparse
import logging
import logging.config
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import elfragmentador

try:
    from argparse import BooleanOptionalAction
except ImportError:
    # Exception for py <3.8 compatibility ...
    BooleanOptionalAction = "store_true"


import warnings

import pandas as pd
import pytorch_lightning as pl
import torch

from elfragmentador import rt
from elfragmentador.datasets import Predictor
from elfragmentador.datasets.peptide_dataset import PeptideDataset
from elfragmentador.datasets.percolator import MokapotPSMDataset, append_preds
from elfragmentador.datasets.sequence_dataset import FastaDataset, SequenceDataset
from elfragmentador.model import PepTransformerModel
from elfragmentador.spectra import SptxtReader
from elfragmentador.train import build_train_parser, main_train

DEFAULT_LOGGER_BASIC_CONF = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": logging.INFO,
}


def _common_checkpoint_args(parser):
    """
    Adds the common to handle model checkpoints to a parser.
    """
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=elfragmentador.DEFAULT_CHECKPOINT,
        help=(
            "Model checkpoint to use for the prediction, "
            "if nothing is passed will download a pretrained model"
        ),
    )
    parser.add_argument(
        "--threads",
        default=2,
        type=int,
        help="Number of threads to use during inference",
    )


def _setup_model(args):
    torch.set_num_threads(args.threads)
    try:
        logging.info(f"Loading model from {args.model_checkpoint}")
        model = PepTransformerModel.load_from_checkpoint(args.model_checkpoint)
    except RuntimeError as e:
        if "Missing key(s) in state_dict":
            logging.error(e)
            logging.error(
                (
                    "Attempting to go though with a bad model,"
                    " DO NOT TRUST THESE RESULTS, "
                    "try to change the checkpoint"
                )
            )
            raise RuntimeError(
                (
                    "The provided checkpoint does not match the version of"
                    " the library, make sure you use a compatible checkpoint"
                    " or you update/downgrade the library"
                )
            )
        else:
            raise RuntimeError(e)

    model.eval()
    return model


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
    Predictor.add_predictor_args(parser)
    return parser


@_gen_cli_help(_append_prediction_parser())
def append_predictions():
    """
    Appends the cosine similarity between the predicted and actual spectra to a
    percolator input.
    """
    log_conf = DEFAULT_LOGGER_BASIC_CONF.copy()
    log_conf.update({"level": logging.INFO})
    logging.basicConfig(**log_conf)
    greeting()

    parser = _append_prediction_parser()
    args = parser.parse_args()
    model = _setup_model(args)
    predictor = Predictor.from_argparse_args(args)

    out_df = append_preds(
        in_pin=args.pin, out_pin=args.out, model=model, predictor=predictor
    )
    logging.info(out_df)


def _predict_csv_parser():
    parser = ArgumentParser(
        prog="elfragmentador_predict_csv", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=str,
        help=(
            "Input csv file,"
            " Expects the csv to have the columns"
            " modified_sequence, collision_energy, precursor_charge OR"
            " 'Modified Sequence', 'CE', 'Precursor Charge'"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output .sptxt file",
    )
    _common_checkpoint_args(parser)
    Predictor.add_predictor_args(parser)
    return parser


@_gen_cli_help(_predict_csv_parser())
def predict_csv():
    """
    Predicts the peptides in a csv file.
    """
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()

    parser = _predict_csv_parser()
    args = parser.parse_args()

    model = _setup_model(args)
    predictor = Predictor.from_argparse_args(args)
    ds = SequenceDataset.from_csv(args.csv)

    ds.predict(model=model, predictor=predictor, batch_size=args.batch_size)
    ds.generate_sptxt(args.out)


def _predict_fasta_parser():
    parser = ArgumentParser(
        prog="elfragmentador_predict_fasta",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta",
        type=str,
        help="Input fasta file",
    )
    parser.add_argument(
        "--enzyme",
        type=str,
        help="Enzyme to use to digest the fasta file",
        default="trypsin",
    )
    parser.add_argument(
        "--nce",
        type=str,
        help="Comma delimited series of collision energies to use",
        default="27,30",
    )
    parser.add_argument(
        "--charges",
        type=str,
        help="Comma delimited series of charges to use",
        default="2,3",
    )
    parser.add_argument(
        "--missed_cleavages",
        type=int,
        help="Maximum number of missed clevages",
        default=2,
    )
    parser.add_argument(
        "--min_length", type=int, help="Minimum peptide length to consider", default=5
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output .sptxt file",
    )
    _common_checkpoint_args(parser)
    Predictor.add_predictor_args(parser)
    return parser


@_gen_cli_help(_predict_fasta_parser())
def predict_fasta():
    """
    Predicts the peptides in a fasta file.
    """
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()

    parser = _predict_fasta_parser()
    args = parser.parse_args()

    model = _setup_model(args)
    predictor = Predictor.from_argparse_args(args)
    nces = [float(x) for x in args.nce.split(",")]
    charges = [float(x) for x in args.charges.split(",")]
    ds = FastaDataset(
        fasta_file=args.fasta,
        enzyme=args.enzyme,
        missed_cleavages=args.missed_cleavages,
        min_length=args.min_length,
        collision_energies=nces,
        charges=charges,
    )

    ds.predict(model=model, predictor=predictor, batch_size=args.batch_size)
    ds.generate_sptxt(args.out)


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
    """
    convert_sptxt Provides a CLI to convert an sptxt to a csv for training.

    provides a CLI for the sptxt_to_csv function, chek that guy out for
    the actual implementation
    """
    logging.basicConfig(**DEFAULT_LOGGER_BASIC_CONF)
    greeting()

    parser = _convert_sptxt_parser()
    args = parser.parse_args()

    logging.info([x.name for x in args.file])

    # Here we make the partial function that will be used to actually read the data
    def converter(fname, outname):
        out = SptxtReader(fname).to_csv(
            outname,
            filter_irt_peptides=args.keep_irts,
            min_delta_ascore=args.min_delta_ascore,
            min_peaks=args.min_peaks,
        )
        return out

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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Path to a file to use as a reference for the evaluation"
            " (.sptxt generally)"
        ),
    )
    parser.add_argument(
        "--screen_nce",
        type=str,
        help="Comma delimited series of collision energies to use",
    )
    parser.add_argument(
        "--max_spec",
        default=1e6,
        type=int,
        help="Maximum number of spectra to read",
    )
    parser.add_argument(
        "--out_csv", type=str, help="Optional csv file to output results to"
    )
    _common_checkpoint_args(parser)
    parser = Predictor.add_argparse_args(parser)
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
    predictor = Predictor.from_argparse_args(args)
    model = _setup_model(args)

    input_file = str(dict_args["input"])
    if input_file.endswith("csv"):
        ds = PeptideDataset.from_csv(
            input_file,
            max_spec=args.max_spec,
            filter_df=False,
            keep_df=True,
        )
    elif input_file.endswith("sptxt") or input_file.endswith("mgf"):
        ds = PeptideDataset.from_sptxt(
            input_file,
            max_spec=args.max_spec,
            filter_df=False,
            keep_df=True,
            min_peaks=0,
            min_delta_ascore=0,
            enforce_length=False,
            pad_zeros=False,
        )
    elif input_file.endswith("feather"):
        ds = PeptideDataset.from_feather(
            input_file,
            max_spec=args.max_spec,
            filter_df=False,
            keep_df=True,
        )
    elif input_file.endswith(".psms.txt"):
        ds = MokapotPSMDataset(
            in_path=input_file,
        )
    else:
        msg = "Input file should have the extension .feather, .csv, .mgf or .sptxt"
        raise ValueError(msg)

    if dict_args["out_csv"] is None:
        logging.warning(
            "No output will be generated because the out_csv argument was not passed"
        )

    if dict_args["screen_nce"] is not None:
        nces = [float(x) for x in dict_args["screen_nce"].split(",")]
    else:
        nces = False

    outs = predictor.evaluate_dataset(
        dataset=ds,
        model=model,
        optimize_nce=nces,
        plot=True,
        keep_predictions=True,
        save_prefix=dict_args["out_csv"],
    )

    summ_out = {"median_" + k: v.median() for k, v in outs._asdict().items()}

    res = pd.DataFrame()
    for k, v in outs._asdict().items():
        res[k] = [x.squeeze().numpy().tolist() for x in v]

    logging.info(summ_out)

    if dict_args["out_csv"] is not None:
        logging.info(f"Writting results to {dict_args['out_csv']}")
        res.to_csv(dict_args["out_csv"], index=False)


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

    logging.info(mod.__repr__().replace("\n", "\t\n"))
    if args.from_checkpoint is not None:
        logging.info(f">> Resuming training from checkpoint {args.from_checkpoint} <<")
        weights_mod = PepTransformerModel.load_from_checkpoint(args.from_checkpoint)
        mod.load_state_dict(weights_mod.state_dict())
        del weights_mod

    main_train(mod, args)
