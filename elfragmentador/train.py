import math
from argparse import ArgumentParser, Namespace
from typing import Union

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

import elfragmentador as ef
from elfragmentador.data import datamodules
from elfragmentador.model import PepTransformerModel


def add_train_parser_args(parser) -> ArgumentParser:
    program_parser = parser.add_argument_group(
        "Program Parameters",
        "Program level parameters, these should not change the outcome of the run",
    )
    model_parser = parser.add_argument_group(
        "Model Parameters",
        "Parameters that modify the model or its training "
        + "(learn rate, scheduler, layers, dimension ...)",
    )
    data_parser = parser.add_argument_group(
        "Data Parameters", "Parameters for the loading of data"
    )
    trainer_parser = parser.add_argument_group(
        "Trainer Parameters", "Parameters that modify the model or its training"
    )

    # add PROGRAM level args
    program_parser.add_argument(
        "--run_name",
        type=str,
        default="ElFragmentador",
        help="Name to be given to the run (logging)",
    )
    program_parser.add_argument(
        "--wandb_project",
        type=str,
        default="elfragmentador",
        help="Wandb project to log to, check out wandb... please",
    )
    trainer_parser.add_argument(
        "--terminator_patience",
        type=int,
        default="5",
        help="Patience for early termination",
    )
    trainer_parser.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="The path of a checkpoint to copy weights from before training",
    )

    # add model specific args
    model_parser = PepTransformerModel.add_model_specific_args(model_parser)

    # add data specific args
    data_parser = datamodules.TrainingDataModule.add_model_specific_args(data_parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    t_parser = pl.Trainer.add_argparse_args(parser)

    if torch.cuda.is_available():
        t_parser.set_defaults(gpus=-1)
        t_parser.set_defaults(precision=16)

    return parser


def get_callbacks(
    run_name: str, termination_patience: int = 20, wandb_project: str = "rttransformer"
) -> dict[
    str,
    Union[
        WandbLogger, list[Union[LearningRateMonitor, ModelCheckpoint, EarlyStopping]]
    ],
]:
    complete_run_name = f"{ef.__version__}_{run_name}"
    wandb_logger = WandbLogger(
        complete_run_name, project=wandb_project, log_model="all"
    )
    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor()
    checkpointer = pl.callbacks.ModelCheckpoint(
        monitor="val_l",
        verbose=True,
        save_top_k=2,
        save_weights_only=True,
        dirpath=".",
        save_last=True,
        mode="min",
        filename=complete_run_name + "_{val_l:.6f}_{epoch:03d}",
    )

    terminator = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_l",
        min_delta=0.00,
        patience=termination_patience,
        verbose=False,
        mode="min",
    )

    return {"logger": wandb_logger, "callbacks": [lr_monitor, checkpointer, terminator]}


def main_train(model: PepTransformerModel, args: Namespace) -> None:
    # TODO add loggging levela and a more structured logger ...
    logger.info(model)
    datamodule = datamodules.TrainingDataModule(
        batch_size=args.batch_size,
        base_dir=args.data_dir,
    )
    datamodule.setup("train")
    spe = math.ceil(datamodule.len_train / datamodule.batch_size)
    logger.info(f">>> TRAIN: Setting steps per epoch to {spe}")
    model.steps_per_epoch = spe

    callbacks = get_callbacks(
        run_name=args.run_name,
        termination_patience=args.terminator_patience,
        wandb_project=args.wandb_project,
    )

    # callbacks["logger"].watch(model.encoder)
    # callbacks["logger"].watch(model.decoder)
    # callbacks["logger"].watch(model.irt_decoder)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=callbacks["logger"],
        callbacks=callbacks["callbacks"],
    )

    model.summarize(max_depth=4)
    model.trainer = trainer
    model.plot_scheduler_lr()
    model.configure_optimizers()

    trainer.fit(model, datamodule)
