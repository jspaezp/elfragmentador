try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version("elfragmentador")

DEFAULT_CHECKPOINT = "https://github.com/jspaezp/elfragmentador/releases/download/v0.44.0/0.44.0a0_onecycle_5e_B_val_l.0.031175_epoch.004.ckpt"
