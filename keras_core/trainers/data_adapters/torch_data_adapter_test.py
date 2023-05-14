import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from keras_core import testing
from keras_core.trainers.data_adapters.torch_data_adapter import (
    TorchDataLoaderAdapter,
)


class SampleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class TestTorchDataLoaderAdapter(testing.TestCase):
    def test_basic_dataloader(self):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        base_ds = SampleDataset(x=x, y=y)
        base_dataloader = DataLoader(base_ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(base_dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, np.ndarray))
            self.assertTrue(isinstance(by, np.ndarray))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))

        ds = adapter.get_torch_dataloader()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, torch.Tensor))
            self.assertTrue(isinstance(by, torch.Tensor))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, torch.float32)
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

    def test_with_torchdataset(self):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))

        base_ds = TensorDataset(x, y)
        base_dataloader = DataLoader(base_ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(base_dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, np.ndarray))
            self.assertTrue(isinstance(by, np.ndarray))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))

        ds = adapter.get_torch_dataloader()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, torch.Tensor))
            self.assertTrue(isinstance(by, torch.Tensor))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, torch.float32)
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))
