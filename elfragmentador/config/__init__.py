from functools import lru_cache
from importlib import resources

from ms2ml.config import Config as MS2MLConfig


@lru_cache(1)
def get_default_config():
    with resources.path("elfragmentador.config", "default_config.toml") as f:
        return MS2MLConfig.from_toml(f)


CONFIG = get_default_config()
CID_CONFIG = MS2MLConfig(CONFIG.asdict().update({ 'g_tolerances': [
    5,
    0.6,
]}))
