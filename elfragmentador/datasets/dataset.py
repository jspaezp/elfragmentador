import logging
from torch.utils.data import Dataset, IterableDataset


class NCEOffsetHolder:
    @property
    def nce_offset(self):
        return self._nce_offset

    @nce_offset.setter
    def nce_offset(self, value):
        logging.warning(f"Setting nce offset to {value}, removing nce overwritting")
        self._nce_offset = value
        self._overwrite_nce = None

    @property
    def overwrite_nce(self):
        return self._overwrite_nce

    @overwrite_nce.setter
    def overwrite_nce(self, value):
        logging.warning(f"Setting nce overwritting to {value}, removing nce offset")
        self._overwrite_nce = value
        self._nce_offset = None

    def calc_nce(self, value):
        if hasattr(self, "overwrite_nce") and self.overwrite_nce:
            value = self.overwrite_nce
        elif hasattr(self, "nce_offset") and self.nce_offset:
            value = value + self.nce_offset

        return value

    def optimize_nce(self):
        raise NotImplementedError


class DatasetBase(Dataset, NCEOffsetHolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDatasetBase(IterableDataset, NCEOffsetHolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __iter__(self):
        raise NotImplementedError
