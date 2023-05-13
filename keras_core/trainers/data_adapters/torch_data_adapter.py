import torch
from tensorflow import nest

from keras_core.trainers.data_adapters.data_adapter import DataAdapter


class TorchDatasetAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataset, class_weight=None):
        if not isinstance(dataset, torch.utils.data.DataLoader):
            raise ValueError(
                f"Expected argument `dataset` to be an instance of"
                f"`torch.utils.data.DataLoader`. Received: {dataset}"
            )
        self._batch_size = dataset.batch_size
        self._size = len(dataset)
        self._partial_batch_size = (
            self._size * self._batch_size - dataset.sampler.num_samples
        )
        ## TODO:
        ## Map class weights

    def get_numpy_iterator(self):
        for batch in self._dataset:
            yield nest.map_structure(lambda x: x.numpy(), batch)

    @property
    def num_batches(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        return self._partial_batch_size > 0

    @property
    def partial_batch_size(self):
        return self._partial_batch_size
