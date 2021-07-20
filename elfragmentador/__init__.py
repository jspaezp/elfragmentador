try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)
DEFAULT_CHECKPOINT = "viz_app/onecycle_5e_petite=0_v_l=0.027239_epoch=004.ckpt"
