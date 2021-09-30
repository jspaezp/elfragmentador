import os
import pytest
from elfragmentador import train, model

cli_commands = [
    ("elfragmentador_train"),
    ("elfragmentador_evaluate"),
    ("elfragmentador_convert_sptxt"),
    ("elfragmentador_calculate_irt"),
    ("elfragmentador_append_pin"),
    ("elfragmentador_predict_csv"),
    ("elfragmentador_predict_fasta"),
]


@pytest.mark.parametrize(
    "command_name",
    cli_commands,
)
def test_cli_help(command_name):
    exit_code = os.system(f"{command_name} --help")
    assert exit_code == 0


args = [
    "--max_epochs 2 --d_model 64 --nhid 120 --nhead 2",
    "--max_epochs 2 --d_model 64 --nhid 120 --nhead 2 --max_spec 20",
]


@pytest.mark.parametrize("arguments", args)
def test_cli_train(shared_datadir, arguments):
    # Set up wandb to pretend that we are already logged in.
    # On a real world setting we would use $ wandb login
    import wandb

    wandb.init(mode="offline")
    wandb.login(anonymous="true", key="A" * 40)

    # Actual cli call
    parser = train.build_train_parser()
    args = parser.parse_args(
        [
            *arguments.split(),
            "--data_dir",
            str(shared_datadir / "train_data_sample"),
        ]
    )
    dict_args = vars(args)
    for k, v in dict_args.items():
        print(f">> {k}: {v}")

    mod = model.PepTransformerModel(**dict_args)
    train.main_train(mod, args)


@pytest.mark.parametrize(
    "csv", ["sample_prediction_csv_2.csv", "sample_prediction_csv_2.csv"]
)
def test_prediction_csv_cli(shared_datadir, csv, tmp_path, checkpoint):
    csv_path = (shared_datadir / "prediction_csv") / csv
    outfile = tmp_path / "foo.sptxt"
    print(csv_path)
    exit_code = os.system(
        f"elfragmentador_predict_csv --model_checkpoint {checkpoint} --csv {csv_path} --out {outfile}"
    )

    assert exit_code == 0

    with open(outfile, "r") as f:
        contents = list(f)

    # print("".join(contents))

    assert len(contents) > 0


def test_evaluation_on_dataset_cli(shared_datadir, checkpoint, tmp_path):
    data = str(shared_datadir) + "/small_phospho_spectrast.sptxt"
    outfile = tmp_path / "foo.csv"

    exit_code = os.system(
        f"elfragmentador_evaluate --model_checkpoint {checkpoint} --input {data} --out_csv {outfile} --screen_nce 1,2,3"
    )

    assert exit_code == 0

    with open(outfile, "r") as f:
        contents = list(f)

    print("".join(contents))

    assert len(contents) > 0


def test_fasta_prediction_cli(shared_datadir, checkpoint, tmp_path):
    fasta_file = (
        shared_datadir / "fasta/uniprot-proteome_UP000464024_reviewed_yes.fasta"
    )
    outfile = tmp_path / "foo.sptxt"

    exit_code = os.system(
        f"elfragmentador_predict_fasta --nce 27 --charges 2,3 --model_checkpoint {checkpoint} --fasta {fasta_file} --out {outfile}"
    )

    assert exit_code == 0

    with open(outfile, "r") as f:
        contents = list(f)

    assert len(contents) > 1


def test_fasta_prediction_cli_variable_batch_size(shared_datadir, checkpoint, tmp_path):
    fasta_file = (
        shared_datadir / "fasta/uniprot-proteome_UP000464024_reviewed_yes.fasta"
    )
    outfile = tmp_path / "foo.sptxt"

    exit_code = os.system(
        f"elfragmentador_predict_fasta --nce 27 --charges 2,3 --model_checkpoint {checkpoint} --fasta {fasta_file} --out {outfile} --batch_size 200"
    )

    assert exit_code == 0

    with open(outfile, "r") as f:
        contents = list(f)

    assert len(contents) > 0
