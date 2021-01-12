from argparse import ArgumentParser

import pytorch_lightning as pl

from transprosit import model
from transprosit import datamodules
from transprosit import train

pl.seed_everything(2020)

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--wandb_project', type=str, default='some_name')

# add model specific args
parser = model.PepTransformerModel.add_model_specific_args(parser)

# add data specific args
parser = datamodules.PeptideDataModule.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

model = model.PepTransformerModel(args)
datamodule = datamodules.PeptideDataModule(args)
datamodule.setup()

callbacks = train.get_callbacks()
trainer = pl.Trainer.from_argparse_args(
    args,
    max_epochs=100,
    precision=16,
    gpus=1,
    profiler="simple",
    logger=callbacks['logger'],
    callbacks=callbacks['callbacks'],
    progress_bar_refresh_rate=50,
)

trainer.fit(model, datamodule)
