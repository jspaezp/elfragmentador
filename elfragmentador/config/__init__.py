from functools import lru_cache
from importlib import resources

from ms2ml.config import Config


@lru_cache(1)
def get_default_config():
    with resources.path("elfragmentador.config", "default_config.toml") as f:
        return Config.from_toml(f)
