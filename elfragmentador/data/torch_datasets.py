from torch.utils.data import DataLoader, TensorDataset


class TupleTensorDataset(TensorDataset):
    def __init__(self, tensor_tuple):
        super().__init__(*tensor_tuple)
        self.builder = type(tensor_tuple)

    def __getitem__(self, index):
        out = self.builder(*super().__getitem__(index))
        return out

    def as_dataloader(self, batch_size, shuffle, num_workers=0, *args, **kwargs):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs,
        )
