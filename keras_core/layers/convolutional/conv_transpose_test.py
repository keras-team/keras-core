import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


class ConvTransposeBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 4),
            "output_shape": (2, 16, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1,),
            "input_shape": (2, 8, 4),
            "output_shape": (2, 23, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 4),
            "output_shape": (2, 16, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv1d_transpose_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv1DTranspose,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "output_padding": output_padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 8, 4),
            "output_shape": (2, 16, 16, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "input_shape": (2, 8, 8, 4),
            "output_shape": (2, 23, 23, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2, 3),
            "strides": (2, 1),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_first",
            "dilation_rate": (1, 1),
            "input_shape": (2, 4, 8, 8),
            "output_shape": (2, 6, 16, 10),
        },
        {
            "filters": 2,
            "kernel_size": (7, 7),
            "strides": (16, 16),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "input_shape": (1, 14, 14, 2),
            "output_shape": (1, 224, 224, 2),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv2d_transpose_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        if (
            data_format == "channels_first"
            and backend.backend() == "tensorflow"
        ):
            pytest.skip("channels_first unsupported on CPU with TF")

        self.run_layer_test(
            layers.Conv2DTranspose,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "output_padding": output_padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
            "input_shape": (2, 8, 8, 8, 4),
            "output_shape": (2, 16, 16, 16, 5),
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
            "input_shape": (2, 8, 8, 8, 4),
            "output_shape": (2, 23, 23, 23, 6),
        },
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
            "input_shape": (2, 8, 8, 8, 4),
            "output_shape": (2, 16, 9, 17, 6),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_conv3d_transpose_basic(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv3DTranspose,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "output_padding": output_padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_bad_init_args(self):
        # `filters` is not positive.
        with self.assertRaises(ValueError):
            layers.Conv1DTranspose(filters=0, kernel_size=1)

        # `kernel_size` has 0.
        with self.assertRaises(ValueError):
            layers.Conv2DTranspose(filters=2, kernel_size=(1, 0))

        # `strides` has 0.
        with self.assertRaises(ValueError):
            layers.Conv2DTranspose(
                filters=2, kernel_size=(2, 2), strides=(1, 0)
            )

        # `dilation_rate > 1` while `strides > 1`.
        with self.assertRaises(ValueError):
            layers.Conv2DTranspose(
                filters=2, kernel_size=(2, 2), strides=2, dilation_rate=(2, 1)
            )


class ConvTransposeCorrectnessTest(testing.TestCase, parameterized.TestCase):

    def _deconv_output_length(
        self,
        input_length,
        filter_size,
        padding,
        output_padding,
        stride,
    ):
        """Determines output length of a transposed convolution given input length."""
        assert padding in {"same", "valid", "full"}

        # Infer length if output padding is None, else compute the exact length
        if output_padding is None:
            if padding == "valid":
                length = input_length * stride + max(filter_size - stride, 0)
            elif padding == "full":
                length = input_length * stride - (stride + filter_size - 2)
            elif padding == "same":
                length = input_length * stride

        else:
            if padding == "same":
                pad = filter_size // 2
            elif padding == "valid":
                pad = 0
            elif padding == "full":
                pad = filter_size - 1

            length = (
                (input_length - 1) * stride + filter_size - 2 * pad + output_padding
            )
        return length
    
    def np_conv2d_transpose(self, x, kernel_weights, bias_weights, strides, padding, output_padding, data_format, dilation_rate):
        if data_format == "channels_first":
            x = x.transpose((0, 2, 3, 1))
        if isinstance(strides, (tuple, list)):
            h_stride, w_stride = strides
        else:
            h_stride = strides
            w_stride = strides
        if isinstance(dilation_rate, (tuple, list)):
            h_dilation, w_dilation = dilation_rate
        else:
            h_dilation = dilation_rate
            w_dilation = dilation_rate
        h_kernel, w_kernel, ch_out, ch_in = kernel_weights.shape

        if h_dilation > 1 or w_dilation > 1:
            new_h_kernel = h_kernel + (h_dilation - 1) * (h_kernel - 1)
            new_w_kernel = w_kernel + (w_dilation - 1) * (w_kernel - 1)
            new_kenel_size_tuple = (new_h_kernel, new_w_kernel)
            new_kernel_weights = np.zeros(
                (*new_kenel_size_tuple, ch_out, ch_in),
                dtype=kernel_weights.dtype,
            )
            new_kernel_weights[::h_dilation, ::w_dilation] = kernel_weights
            kernel_weights = new_kernel_weights
            h_kernel, w_kernel = kernel_weights.shape[:2]

        n_batch, h_x, w_x, _ = x.shape
        # Define output shape to max shape
        h_out = self._deconv_output_length(h_x, h_kernel, padding="valid", output_padding=None, stride=h_stride)
        w_out = self._deconv_output_length(w_x, w_kernel, padding="valid", output_padding=None, stride=w_stride)

        # h_out = (h_x - 1) * h_stride + h_kernel + output_padding
        # w_out = (w_x - 1) * w_stride + w_kernel + output_padding
        
        # Compute output
        output = np.zeros([n_batch, h_out, w_out, ch_out])
        for nb in range(n_batch):
            for h_x_idx in range(h_x):
                h_out_idx = h_x_idx * h_stride # Index in output
                for w_x_idx in range(w_x):
                    w_out_idx = w_x_idx * w_stride
                    output[nb, h_out_idx: h_out_idx + h_kernel, w_out_idx: w_out_idx + w_kernel, :] += np.sum(kernel_weights[:, :, :, :] *
                                                                                            x[nb, h_x_idx, w_x_idx, :], axis=-1)
        output = output + bias_weights
        # Crop output to the expected output shape
        h_expected = self._deconv_output_length(h_x, h_kernel, padding, output_padding, h_stride)
        w_expected = self._deconv_output_length(w_x, w_kernel, padding, output_padding, w_stride)
        h_pad = h_out - h_expected
        w_pad = w_out - w_expected
        output = output[:, :h_out-h_pad, :w_out-w_pad]
        return output

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1,),
        },
        {
            "filters": 6,
            "kernel_size": (2,),
            "strides": (2,),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
    )
    def test_conv1d_transpose(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.Conv1DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.Conv1DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-5)

    @parameterized.parameters(
        # {
        #     "filters": 5,
        #     "kernel_size": 2,
        #     "strides": 2,
        #     "padding": "valid",
        #     "output_padding": None,
        #     "data_format": "channels_last",
        #     "dilation_rate": 1,
        # },
        {
            "filters": 6,
            "kernel_size": 7,
            "strides": 16,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
        },
        # {
        #     "filters": 6,
        #     "kernel_size": (2, 3),
        #     "strides": (2, 1),
        #     "padding": "valid",
        #     "output_padding": None,
        #     "data_format": "channels_last",
        #     "dilation_rate": (1, 1),
        # },
        # {
        #     "filters": 2,
        #     "kernel_size": (7, 7),
        #     "strides": (16, 16),
        #     "padding": "valid",
        #     "output_padding": None,
        #     "data_format": "channels_last",
        #     "dilation_rate": (1, 1),
        # },
    )
    def test_conv2d_transpose(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        # tf_keras_layer = tf.keras.layers.Conv2DTranspose(
        #     filters=filters,
        #     kernel_size=kernel_size,
        #     strides=strides,
        #     padding=padding,
        #     output_padding=output_padding,
        #     data_format=data_format,
        #     dilation_rate=dilation_rate,
        # )

        inputs = np.random.normal(size=[2, 14, 14, 4])
        layer.build(input_shape=inputs.shape)
        # tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        # tf_keras_layer.kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        # tf_keras_layer.bias.assign(bias_weights)

        outputs = layer(inputs)
        # expected = tf_keras_layer(inputs)
        expected = self.np_conv2d_transpose(inputs, kernel_weights, bias_weights, strides, padding, output_padding,data_format, dilation_rate)
        self.assertAllClose(outputs, expected, atol=1e-5)

    @parameterized.parameters(
        {
            "filters": 5,
            "kernel_size": 2,
            "strides": 2,
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": 1,
        },
        {
            "filters": 6,
            "kernel_size": 2,
            "strides": 3,
            "padding": "same",
            "output_padding": 2,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
        },
        {
            "filters": 6,
            "kernel_size": (2, 2, 3),
            "strides": (2, 1, 2),
            "padding": "valid",
            "output_padding": None,
            "data_format": "channels_last",
            "dilation_rate": (1, 1, 1),
        },
    )
    def test_conv3d_transpose(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    ):
        layer = layers.Conv3DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )
        tf_keras_layer = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

        inputs = np.random.normal(size=[2, 8, 8, 8, 4])
        layer.build(input_shape=inputs.shape)
        tf_keras_layer.build(input_shape=inputs.shape)

        kernel_shape = layer.kernel.shape
        kernel_weights = np.random.normal(size=kernel_shape)
        bias_weights = np.random.normal(size=(filters,))
        layer.kernel.assign(kernel_weights)
        tf_keras_layer.kernel.assign(kernel_weights)

        layer.bias.assign(bias_weights)
        tf_keras_layer.bias.assign(bias_weights)
        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-5)
