import pytest
from transprosit import train

def test_cli_help():
    parser = train.build_parser()
    # Just checks that I did not break the parser in any other script...
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parser.parse_args(["--help"])

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0