from pathlib import Path
import argparse
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
import warnings

import pytorch_lightning as pl

from elfragmentador.train import build_train_parser, main_train
from elfragmentador.model import PepTransformerModel
from elfragmentador.spectra import sptxt_to_csv
from elfragmentador import evaluate


def convert_sptxt():
    parser = ArgumentParser()
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        nargs="+",
        help="Input file(s) to convert (sptxt)",
    )
    parser.add_argument("--warn", default=False, action=BooleanOptionalAction, help="Wether to show warnings or not")
    parser.add_argument("--filter_irts", default=True, action=BooleanOptionalAction, help="Wether to remove sequences that match procal and biognosys iRT peptides")

    args = parser.parse_args()

    print([x.name for x in args.file])

    for f in args.file:
        out_file = f.name + ".csv"
        if Path(out_file).exists():
            print(
                f"Skipping conversion of '{f.name}' to '{out_file}',"
                " because {out_file} exists."
            )
        else:
            print(f"Converting '{f.name}' to '{out_file}'")
            if args.warn:
                sptxt_to_csv(f.name, out_file, filter_irt_peptides=args.filter_irts)
            else:
                with warnings.catch_warnings():
                    sptxt_to_csv(f.name, out_file, filter_irt_peptides=args.filter_irts)


def evaluate_checkpoint():
    pl.seed_everything(2020)
    parser = evaluate.build_evaluate_parser()
    args = parser.parse_args()
    dict_args = vars(args)

    evaluate.evaluate_checkpoint(**dict_args)


def train():
    pl.seed_everything(2020)
    parser = build_train_parser()
    args = parser.parse_args()
    dict_args = vars(args)
    print("\n====== Passed command line args/params =====\n")
    for k, v in dict_args.items():
        print(f">> {k}: {v}")

    mod = PepTransformerModel(**dict_args)
    if args.from_checkpoint is not None:
        print(f"\n>> Resuming training from checkpoint {args.from_checkpoint} <<\n")
        weights_mod = PepTransformerModel.load_from_checkpoint(args.from_checkpoint)
        mod.load_state_dict(weights_mod.state_dict())
        del weights_mod

    main_train(mod, args)
