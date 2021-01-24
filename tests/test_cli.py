import pytest
from transprosit import train, model


def test_cli_help():
    parser = train.build_parser()
    # Just checks that I did not break the parser in any other script...
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parser.parse_args(["--help"])

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_cli_train(shared_datadir):
    parser = train.build_parser()
    args = parser.parse_args(["--fast_dev_run", "1", "--data_dir", str(shared_datadir)])
    dict_args = vars(args)
    for k, v in dict_args.items():
        print(f">> {k}: {v}")

    mod = model.PepTransformerModel(**dict_args)
    train.main_train(mod, args)
