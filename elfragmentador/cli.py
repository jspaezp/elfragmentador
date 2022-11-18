import dataclasses
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pytorch_lightning as pl
import torch
from loguru import logger
from ms2ml.data.parsing.pin import comet_pin_to_df

import elfragmentador
from elfragmentador.config import CONFIG
from elfragmentador.data.predictor import Predictor
from elfragmentador.model import PepTransformerModel
from elfragmentador.train import add_train_parser_args, main_train


def common_checkpoint_args(parser):
    """Adds the common to handle model checkpoints to a parser."""
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


def setup_model(args):
    torch.set_num_threads(args.threads)
    try:
        logger.info(f"Loading model from {args.model_checkpoint}")
        if args.model_checkpoint == "RANDOM":
            logger.info("Using random model")
            model = PepTransformerModel(
                combine_embeds=True,
                num_decoder_layers=3,
                num_encoder_layers=2,
                nhid=48 * 2,
                d_model=48,
            )
        else:
            model = PepTransformerModel.load_from_checkpoint(args.model_checkpoint)
    except RuntimeError as e:
        if "Missing key(s) in state_dict":
            logger.error(e)
            logger.error(
                "Attempting to go though with a bad model,"
                " DO NOT TRUST THESE RESULTS, "
                "try to change the checkpoint"
            )
            raise RuntimeError(
                "The provided checkpoint does not match the version of"
                " the library, make sure you use a compatible checkpoint"
                " or you update/downgrade the library"
            )
        else:
            raise RuntimeError(e)

    model = model.eval()
    return model


def greeting():
    logger.info(f"ElFragmentador version: {elfragmentador.__version__}")


def startup_setup(fun):
    def wrapper(*args, **kwargs):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        pl.seed_everything(2020)
        greeting()
        fun(*args, **kwargs)

    return wrapper


def add_pin_append_parser_args(parser):
    parser.add_argument(
        "--pin",
        type=str,
        help="Input percolator file",
    )
    parser.add_argument(
        "--nce",
        type=float,
        help="Collision energy to use for the prediction",
        default="32",
    )
    parser.add_argument(
        "--rawfile_locations",
        type=str,
        help="Locations to look for the raw files",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Input percolator file",
    )
    common_checkpoint_args(parser)
    return parser


@startup_setup
def append_predictions(args):
    """Appends data to a pin file

    Appends the cosine similarity between the predicted and actual spectra to a
    percolator input.
    """
    model = setup_model(args)

    predictor = Predictor(model=model)
    df = predictor.compare(adapter=args.input, nce=args.nce)
    df2 = comet_pin_to_df(args.pin)
    for col in df.columns:
        df.insert(loc=4, column=col, value=df2[col])

    logger.info(df)
    df.to_csv(args.out, sep="\t", index=False)


def add_predict_parser_args(parser):
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
        type=float,
        help="Collision energy to use for the prediction",
        default="32",
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
        help="Output .dlib file",
    )
    common_checkpoint_args(parser)
    return parser


@startup_setup
def predict(args):
    """Predicts the peptides in a fasta file."""
    model = setup_model(args)
    charges = [int(x) for x in args.charges.split(",")]

    config = dataclasses.replace(CONFIG)
    config.charges = charges
    config.peptide_length_range = (args.min_length, config.peptide_length_range[1])
    if args.enzyme != "trypsin":
        ValueError("Only trypsin is supported for now")

    if args.missed_cleavages != 2:
        ValueError("Only 2 missed cleavages is supported for now")

    predictor = Predictor(model=model)
    predictor.predict_to_file(
        adapter=args.fasta, out_filepath=args.out, nce=args.nce, charges=charges
    )


def add_evaluate_parser_args(parser):
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Path to a file to use as a reference for the evaluation (.sptxt generally)"
        ),
    )
    parser.add_argument(
        "--nce",
        type=str,
        help="Comma delimited series of collision energies to use",
    )
    parser.add_argument("--out", type=str, help="csv file to output results to")
    common_checkpoint_args(parser)
    return parser


@startup_setup
def evaluate_checkpoint(args):
    model = setup_model(args)
    nces = [float(x) for x in args.nce.split(",")]

    predictor = Predictor(model=model)
    predictor.compare_to_file(adapter=args.input, out_filepath=args.out, nce=nces)


def train(args):
    dict_args = vars(args)
    logger.info("====== Passed command line args/params =====")
    for k, v in dict_args.items():
        logger.info(f">> {k}: {v}")

    mod = PepTransformerModel(**dict_args)

    logger.info(mod.__repr__().replace("\n", "\t\n"))
    if args.from_checkpoint is not None:
        logger.info(f">> Resuming training from checkpoint {args.from_checkpoint} <<")
        weights_mod = PepTransformerModel.load_from_checkpoint(args.from_checkpoint)
        mod.load_state_dict(weights_mod.state_dict())
        del weights_mod

    main_train(mod, args)


# create the top-level parser

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers()

# create the parser for the "foo" command

parser_append = subparsers.add_parser("append_pin")
add_pin_append_parser_args(parser=parser_append)
parser_append.set_defaults(func=append_predictions)

# create the parser for the "bar" command
parser_evaluate = subparsers.add_parser("evaluate")
add_evaluate_parser_args(parser_evaluate)
parser_evaluate.set_defaults(func=evaluate_checkpoint)

parser_predict = subparsers.add_parser("predict")
add_predict_parser_args(parser=parser_predict)
parser_predict.set_defaults(func=predict)

parser_train = subparsers.add_parser("train")
add_train_parser_args(parser=parser_train)
parser_train.set_defaults(func=train)


def main_cli():
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main_cli()
