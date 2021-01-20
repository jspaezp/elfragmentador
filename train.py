from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl

from transprosit import model
from transprosit import datamodules
from transprosit.train import main_train

pl.seed_everything(2020)

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument(
    "--run_name",
    type=str,
    default=f"prosit_transformer",
    help="Name to be given to the run (logging)",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="rttransformer",
    help="Wandb project to log to, check out wandb... please",
)
parser.add_argument(
    "--terminator_patience",
    type=int,
    default="5",
    help="Patience for early termination",
)

# add model specific args
parser = model.PepTransformerModel.add_model_specific_args(parser)

# add data specific args
parser = datamodules.PeptideDataModule.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)

# This is necessary to add the defaults when calling the help option
parser = ArgumentParser(
    parents=[parser],
    add_help=False,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

def test_cli_help():
    import pytest
    # Just checks that I did not break the parser in any other script...
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parser.parse_args(["--help"])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

if __name__ == "__main__":

    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    model = model.PepTransformerModel(**dict_args)
    main_train(model, args)
