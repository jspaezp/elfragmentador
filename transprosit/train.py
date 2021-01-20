import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transprosit import datamodules


def get_callbacks(run_name, termination_patience=20, wandb_project="rttransformer"):
    wandb_logger = WandbLogger(run_name, project=wandb_project)
    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="epoch")
    checkpointer = pl.callbacks.ModelCheckpoint(
        prefix=run_name,
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

    callbacks = get_callbacks(
        run_name=args.run_name,
        termination_patience=args.terminator_patience,
        wandb_project=args.wandb_project,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=100,
        precision=16,
        gpus=1,
        profiler="simple",
        logger=callbacks["logger"],
        callbacks=callbacks["callbacks"],
        progress_bar_refresh_rate=50,
    )

    trainer.fit(model, datamodule)
