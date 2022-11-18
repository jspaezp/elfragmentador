import sqlite3

import pandas as pd
import pytest

from elfragmentador.cli import comet_pin_to_df, main_cli

cli_args_help = [
    ("predict", "--help"),
    ("train", "--help"),
    ("evaluate", "--help"),
    ("append_pin", "--help"),
]


@pytest.mark.parametrize(
    "args",
    cli_args_help,
)
def test_cli_help(args):
    with pytest.raises(SystemExit) as error:
        main_cli(list(args))

    assert error.value.code == 0


def test_cli_train(shared_datadir):
    # Set up wandb to pretend that we are already logged in.
    # On a real world setting we would use $ wandb login
    import wandb

    wandb.init(mode="offline")

    arguments = (
        "--max_epochs",
        "200",
        "--d_model",
        "64",
        "--nhid",
        "120",
        "--nhead",
        "2",
        "--lr_ratio",
        "50",
        "--scheduler",
        "onecycle",
        "--fast_dev_run",
    )
    # Actual cli call
    args = (
        ["train", "--limit_train_batches", "2"]
        + list(arguments)
        + ["--data_dir", str(shared_datadir / "parquet")]
    )
    main_cli(args)


def test_evaluation_on_dataset_cli(shared_datadir, tmp_path):
    data = str(shared_datadir) + "/dlib/small_yeast.dlib"
    outfile = tmp_path / "foo.csv"

    main_cli(
        [
            "evaluate",
            "--model_checkpoint",
            "RANDOM",
            "--input",
            f"{data}",
            "--out",
            f"{outfile}",
            "--nce",
            "25,30",
        ]
    )

    with open(outfile) as f:
        contents = list(f)

    print("".join(contents))

    assert len(contents) > 0


def test_fasta_prediction_cli(shared_datadir, tmp_path):
    fasta_file = shared_datadir / "fasta/P0DTC4.fasta"
    outfile = tmp_path / "foo.dlib"

    main_cli(
        [
            "predict",
            "--nce",
            "27",
            "--charges",
            "2,3",
            "--model_checkpoint",
            "RANDOM",
            "--fasta",
            f"{str(fasta_file)}",
            "--out",
            f"{outfile}",
        ]
    )

    con = sqlite3.Connection(outfile)
    df = pd.read_sql_query("SELECT * from entries", con)
    con.close()

    assert len(df) > 1


def test_pin_append_cli(shared_datadir, tmp_path):
    pin_file = (
        shared_datadir
        / "01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.pin"
    )
    raw_location = (
        shared_datadir / "/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1"
    )
    outfile = tmp_path / "foo.pin"

    main_cli(
        [
            "append_pin",
            "--nce",
            "27",
            "--model_checkpoint",
            "RANDOM",
            "--pin",
            f"{str(pin_file)}",
            "--rawfile_location",
            f"{str(raw_location)}",
            "--out",
            f"{str(outfile)}",
        ]
    )

    orig_df = comet_pin_to_df(str(pin_file))
    df = comet_pin_to_df(str(outfile))

    assert len(df) == len(orig_df)
    assert len(df.columns) > len(orig_df.columns)
