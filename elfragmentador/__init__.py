try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version("elfragmentador")

DEFAULT_CHECKPOINT = "https://github.com/jspaezp/elfragmentador-modelzoo/raw/main/0.50.0a5_onecycle_5e_C_val_l%3D0.038469_epoch%3D004.ckpt"
