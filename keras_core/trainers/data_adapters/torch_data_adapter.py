from tensorflow import nest

from keras_core.trainers.data_adapters.data_adapter import DataAdapter


class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataloader, class_weight=None):
        import torch

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError(
                f"Expected argument `dataloader` to be an instance of"
                f"`torch.utils.data.DataLoader`. Received: {dataloader}"
            )
        if class_weight:
            raise ValueError(
                "`class_weight` is not supported at "
                "the moment in `TorchDataLoaderAdapter`."
            )

        self._dataloader = dataloader
        self._batch_size = dataloader.batch_size
        self._size = len(dataloader)

        # If DataLoader is created using an instance of `TensorDataset`
        # then the `num_samples` property for the corresponding sampler
        # doesn't exist. In that case we will set partail batch size
        # to `None`.
        try:
            self._partial_batch_size = (
                self._size * self._batch_size - dataloader.sampler.num_samples
            )
        except:
            self._partial_batch_size = None

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
