import numpy as np

from keras_core import testing
from keras_core.backend import KerasTensor
from keras_core.layers import InputLayer


class InputLayerTest(testing.TestCase):
    # Testing happy path for layer without input tensor
    def test_input_basic(self):
        input_shape = (2, 3)
        batch_size = 4
        dtype = "float32"
        ndim = len(tuple((batch_size,) + input_shape))

        values = InputLayer(
            shape=input_shape, batch_size=batch_size, dtype=dtype
        )

        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.batch_shape[0], batch_size)
        self.assertEqual(values.batch_shape[1:], input_shape)
        self.assertEqual(values.trainable, True)
        self.assertIsInstance(values.output, KerasTensor)
        self.assertEqual(values.output.ndim, ndim)
        self.assertEqual(values.output.dtype, dtype)

    # Testing shape is not None and batch_shape is not None condition
    def test_input_error1(self):
        input_shape = (2, 3)

        try:
            InputLayer(shape=input_shape, batch_shape=input_shape)
        except ValueError as x:
            self.assertEqual(
                x.args[0],
                "You cannot pass both `shape` and `batch_shape` at the "
                "same time.",
            )

    # Testing batch_size is not None and batch_shape is not None
    def test_input_error2(self):
        input_shape = (2, 3)
        batch_size = 4

        try:
            InputLayer(batch_size=batch_size, batch_shape=input_shape)
        except ValueError as x:
            self.assertEqual(
                x.args[0],
                "You cannot pass both `batch_size` and `batch_shape` at the "
                "same time.",
            )

    # Testing shape is None and batch_shape is None
    def test_input_error3(self):
        try:
            InputLayer(shape=None, batch_shape=None)
        except ValueError as x:
            self.assertEqual(x.args[0], "You must pass a `shape` argument.")

    # Testing Input tensor is not Keras tensor
    def test_input_tensor_error(self):
        input_shape = (2, 3)
        batch_size = 4
        input_tensor = np.zeros(input_shape)
        try:
            InputLayer(
                shape=input_shape,
                batch_size=batch_size,
                input_tensor=input_tensor,
            )
        except ValueError as x:
            self.assertEqual(
                x.args[0],
                "Argument `input_tensor` must be a KerasTensor. "
                f"Received invalid type: input_tensor={input_tensor} "
                f"(of type {type(input_tensor)})",
            )

    # Testing happy path for layer with input tensor
    def testing_input_tensor(self):
        input_shape = (2, 3)
        batch_size = 4
        dtype = "float32"
        input_tensor = KerasTensor(shape=input_shape, dtype=dtype)

        values = InputLayer(
            shape=input_shape,
            batch_size=batch_size,
            input_tensor=input_tensor,
            dtype=dtype,
        )

        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.batch_shape[0], batch_size)
        self.assertEqual(values.batch_shape[1:], input_shape)
        self.assertEqual(values.trainable, True)
        self.assertIsInstance(values.output, KerasTensor)
        self.assertEqual(values.output, input_tensor)
        self.assertEqual(values.output.ndim, input_tensor.ndim)
        self.assertEqual(values.output.dtype, dtype)
