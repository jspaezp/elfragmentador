from transprosit.datamodules import iRTDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def get_callbacks(run_name, termination_patience=20, wandb_project="rttransformer"):
    wandb_logger = WandbLogger(run_name, project=wandb_project)
    lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="epoch")
    checkpointer = pl.callbacks.ModelCheckpoint(
        prefix=run_name,
        monitor="val_loss",
        verbose=True,
        save_top_k=2,
        save_weights_only=True,
        dirpath=".",
        save_last=True,
        mode="min",
        filename="{val_loss:.6f}_{epoch:03d}",
    )

    terminator = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=termination_patience,
        verbose=False,
        mode="min",
    )

    return {"logger": wandb_logger, "callbacks": [lr_monitor, checkpointer, terminator]}
