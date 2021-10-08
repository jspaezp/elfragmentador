try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version("elfragmentador")

DEFAULT_CHECKPOINT = "https://github.com/jspaezp/elfragmentador-modelzoo/raw/main/0.51.0a/0.51.0a0_Onecycle_15e_48_192_val_l%3D0.155363_epoch%3D014.ckpt"
