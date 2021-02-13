from pathlib import Path
import argparse
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)

try:
    from argparse import BooleanOptionalAction
except ImportError:
    # Exception for py <3.8 compatibility ...
    BooleanOptionalAction = "store_true"


import warnings

import pytorch_lightning as pl

from elfragmentador.train import build_train_parser, main_train
from elfragmentador.model import PepTransformerModel
from elfragmentador.spectra import sptxt_to_csv
from elfragmentador import evaluate, rt


def calculate_irt():
    parser = ArgumentParser()
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

    args = parser.parse_args()
    files = [x.name for x in args.file]
    df = rt.calculate_multifile_iRT(files)
    df.to_csv(str(args.out))


def convert_sptxt():
    parser = ArgumentParser()
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

    args = parser.parse_args()

    print([x.name for x in args.file])
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
            print(
                f"Skipping conversion of '{f.name}' "
                f"to '{out_file}', "
                f"because {out_file} exists."
            )
        else:
            print(f"Converting '{f.name}' to '{out_file}'")
            if args.warn:
                print("Warning stuff ...")
                converter(f.name, out_file)
            else:
                with warnings.catch_warnings(record=True) as c:
                    print("Not Warning stuff ...")
                    converter(f.name, out_file)

                if len(c) > 0:
                    warnings.warn(f"Last Error Message of {len(c)}: '{c[-1].message}'")


def evaluate_checkpoint():
    pl.seed_everything(2020)
    parser = evaluate.build_evaluate_parser()
    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    if dict_args["csv"] is not None:
        model = PepTransformerModel.load_from_checkpoint(args.checkpoint_path)
        evaluate.evaluate_on_csv(
            model,
            args.csv,
            batch_size=args.batch_size,
            device=args.device,
            max_spec=args.max_spec,
        )
    else:
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


if __name__ == "__main__":
    model = PepTransformerModel.load_from_checkpoint(
        "/home/jspaezp/Downloads/0.23.0_onecycle_20e_petite-v_l=0.137555_epoch=014.ckpt"
    )
    evaluate.evaluate_on_csv(
        model,
        "~/Downloads/holdout_combined_massive_20200212.sptxt.irt.sptxt.csv",
        batch_size=4,
        max_spec=5000,
    )
