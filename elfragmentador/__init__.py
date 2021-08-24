try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version("elfragmentador")

DEFAULT_CHECKPOINT = "https://github.com/jspaezp/elfragmentador/releases/download/v0.49.1/0.49.1_onecycle_5e_C_val_l.0.029633_epoch.004.ckpt"
