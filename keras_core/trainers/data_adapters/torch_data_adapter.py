from tensorflow import nest

from keras_core.trainers.data_adapters.data_adapter import DataAdapter


class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataloader):
        import torch

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError(
                f"Expected argument `dataloader` to be an instance of"
                f"`torch.utils.data.DataLoader`. Received: {dataloader}"
            )

        self._dataloader = dataloader
        self._batch_size = dataloader.batch_size
        self._size = len(dataloader)
        self._partial_batch_size = (
                len(dataloader.dataset) % self._batch_size
            )

    def get_numpy_iterator(self):
        for batch in self._dataloader:
            yield nest.map_structure(lambda x: x.numpy(), batch)

    def get_torch_dataloader(self):
        return self._dataloader

    @property
    def num_batches(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        if self._partial_batch_size:
            return self._partial_batch_size > 0
        else:
            return None

    @property
    def partial_batch_size(self):
        return self._partial_batch_size
