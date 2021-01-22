import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transprosit import datamodules, model
import transprosit as tp


def build_parser():
    parser = ArgumentParser(add_help=False)

    program_parser = parser.add_argument_group(
        "Program Parameters",
        "Program level parameters, these should not change the outcome of the run",
    )
    model_parser = parser.add_argument_group(
        "Model Parameters",
        "Parameters that modify the model or its training (learn rate, scheduler, layers, dimension ...)",
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
        default=f"prosit_transformer",
        help="Name to be given to the run (logging)",
    )
    program_parser.add_argument(
        "--wandb_project",
        type=str,
        default="rttransformer",
        help="Wandb project to log to, check out wandb... please",
    )
    trainer_parser.add_argument(
        "--terminator_patience",
        type=int,
        default="5",
        help="Patience for early termination",
    )

    # add model specific args
    model_parser = model.PepTransformerModel.add_model_specific_args(model_parser)

    # add data specific args
    data_parser = datamodules.PeptideDataModule.add_model_specific_args(data_parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    t_parser = ArgumentParser(add_help=False)
    t_parser = pl.Trainer.add_argparse_args(t_parser)
    t_parser.set_defaults(gpus=-1)

    parser = ArgumentParser(
        parents=[t_parser, parser], formatter_class=ArgumentDefaultsHelpFormatter
    )

    return parser


def get_callbacks(run_name, termination_patience=20, wandb_project="rttransformer"):
    complete_run_name = f"{tp.__version__}_{run_name}"
    wandb_logger = WandbLogger(complete_run_name, project=wandb_project)
    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor()
    checkpointer = pl.callbacks.ModelCheckpoint(
        prefix=complete_run_name,
        monitor="v_l",
        verbose=True,
        save_top_k=2,
        save_weights_only=True,
        dirpath=".",
        save_last=True,
        mode="min",
        filename="{v_l:.6f}_{epoch:03d}",
    )

    terminator = pl.callbacks.early_stopping.EarlyStopping(
        monitor="v_l",
        min_delta=0.00,
        patience=termination_patience,
        verbose=False,
        mode="min",
    )

    return {"logger": wandb_logger, "callbacks": [lr_monitor, checkpointer, terminator]}


def main_train(model, args):
    print(model)
    datamodule = datamodules.PeptideDataModule(
        batch_size=args.batch_size, base_dir=args.data_dir
    )
    datamodule.setup()
    spe = math.ceil(len(datamodule.train_dataset) / datamodule.batch_size)
    print(f">>> TRAIN: Setting steps per epoch to {spe}")
    model.steps_per_epoch = spe

    callbacks = get_callbacks(
        run_name=args.run_name,
        termination_patience=args.terminator_patience,
        wandb_project=args.wandb_project,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        profiler="simple",
        logger=callbacks["logger"],
        callbacks=callbacks["callbacks"],
    )

    trainer.fit(model, datamodule)


def main():
    pl.seed_everything(2020)
    parser = build_parser()
    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    mod = model.PepTransformerModel(**dict_args)
    main_train(mod, args)