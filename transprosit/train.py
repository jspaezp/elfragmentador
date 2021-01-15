from transprosit.model import LitTransformer
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


def rt_main(*args, **kwargs):
    pl.seed_everything(2020)
    dm = iRTDataModule(batch_size=48 * 30)

    run_name = f"AdamIRTTransformerV3"
    print(f"\n\n>>> {run_name}\n\n")
    # Number of features must be diviible by the number of heads
    model = LitTransformer(
        dropout=0, nlayers=6, nhead=12, ninp=516, nhid=1024, lr=1e-4, max_len=26
    )
    print(model)

    callbacks = get_callbacks(run_name=run_name)

    trainer = pl.Trainer(
        max_epochs=100,
        precision=16,
        gpus=1,
        profiler="simple",
        logger=callbacks["logger"],
        callbacks=callbacks["callbacks"],
        progress_bar_refresh_rate=50,
    )

    trainer.fit(model, dm)
    callbacks["logger"].finalize("ok")
    callbacks["logger"].experiment.finish(exit_code=0)


if __name__ == "__main__":
    rt_main()
