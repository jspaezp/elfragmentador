import pytest
from elfragmentador import train, model


def test_cli_help():
    parser = train.build_train_parser()
    # Just checks that I did not break the parser in any other script...
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parser.parse_args(["--help"])

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_cli_train(shared_datadir):
    # Set up wandb to pretend that we are already logged in.
    # On a real world setting we would use $ wandb login
    import wandb
    wandb.init(mode="offline")
    wandb.login(anonymous = "true", key="A" * 40)

    # Actual cli call
    parser = train.build_train_parser()
    args = parser.parse_args(["--fast_dev_run", "1", "--data_dir", str(shared_datadir)])
    dict_args = vars(args)
    for k, v in dict_args.items():
        print(f">> {k}: {v}")

    mod = model.PepTransformerModel(**dict_args)
    train.main_train(mod, args)
