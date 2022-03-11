try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version("elfragmentador")

DEFAULT_CHECKPOINT = "https://github.com/jspaezp/elfragmentador-modelzoo/raw/main/0.51.2/0.51.2_onecycle_10e_96_120_val_l%3D0.143788_epoch%3D009.ckpt"  # noqa: E501
