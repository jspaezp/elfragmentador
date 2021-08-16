import os
import pytest
from elfragmentador import train, model

cli_commands = [
    ("elfragmentador_train"),
    ("elfragmentador_evaluate"),
    ("elfragmentador_convert_sptxt"),
    ("elfragmentador_calculate_irt"),
    ("elfragmentador_append_pin"),
]


@pytest.mark.parametrize(
    "command_name",
    cli_commands,
)
def test_cli_help(command_name):
    exit_code = os.system(f"{command_name} --help")
    assert exit_code == 0


def test_cli_train(shared_datadir):
    # Set up wandb to pretend that we are already logged in.
    # On a real world setting we would use $ wandb login
    import wandb

    wandb.init(mode="offline")
    wandb.login(anonymous="true", key="A" * 40)

    # Actual cli call
    parser = train.build_train_parser()
    args = parser.parse_args(
        ["--fast_dev_run", "1", "--data_dir", str(shared_datadir / "train_data_sample")]
    )
    dict_args = vars(args)
    for k, v in dict_args.items():
        print(f">> {k}: {v}")

    mod = model.PepTransformerModel(**dict_args)
    train.main_train(mod, args)
