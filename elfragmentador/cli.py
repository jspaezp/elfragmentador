import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger

import elfragmentador
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

    model.eval()
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
        "--out",
        type=str,
        help="Input percolator file",
    )
    common_checkpoint_args(parser)
    return parser


@startup_setup
def append_predictions(args):
    """
    Appends the cosine similarity between the predicted and actual spectra to a.

    percolator input.
    """
    model = setup_model(args)

    out_df = append_preds(
        in_pin=args.pin, out_pin=args.out, model=model, predictor=predictor
    )
    logger.info(out_df)


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
    common_checkpoint_args(parser)
    return parser


@startup_setup
def predict_csv(args):
    """Predicts the peptides in a csv file."""
    model = setup_model(args)
    # predictor = Predictor.from_argparse_args(args)
    ds = SequenceDataset.from_csv(args.csv)

    ds.predict(model=model, predictor=predictor, batch_size=args.batch_size)
    ds.generate_sptxt(args.out)


@startup_setup
def predict_fasta(args):
    """Predicts the peptides in a fasta file."""
    model = setup_model(args)
    # predictor = Predictor.from_argparse_args(args)
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


def add_evaluate_parser_args(parser):
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Path to a file to use as a reference for the evaluation (.sptxt generally)"
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
    common_checkpoint_args(parser)
    return parser


@startup_setup
def evaluate_checkpoint(args):
    dict_args = vars(args)
    logger.info(dict_args)
    # predictor = Predictor.from_argparse_args(args)
    model = setup_model(args)

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
        logger.warning(
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

    logger.info(summ_out)

    if dict_args["out_csv"] is not None:
        logger.info(f"Writting results to {dict_args['out_csv']}")
        res.to_csv(dict_args["out_csv"], index=False)


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
parser_predict.set_defaults(func=predict_fasta)

parser_train = subparsers.add_parser("train")
add_train_parser_args(parser=parser_train)
parser_train.set_defaults(func=train)

if __name__ == "__main__":
    parser.parse_args()
