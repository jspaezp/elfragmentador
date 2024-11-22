import importlib.metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version("elfragmentador")
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"
