import math

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transprosit import datamodules
import transprosit as tp


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
