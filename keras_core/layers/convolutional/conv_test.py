import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import layers
from keras_core import testing


class ConvBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        (5, 2, 1, "valid", "channels_last", 1, 1, (3, 5, 4), (3, 4, 5)),
        (6, 2, 1, "same", "channels_first", (1,), 2, (3, 4, 4), (3, 6, 4)),
        (6, (2,), (2,), "valid", "channels_last", 1, 2, (3, 5, 4), (3, 2, 6)),
    )
    def test_conv1d(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.Conv1D,
            init_kwargs={
                "filters": filters,
                "kernel_size": kernel_size,
                "strides": strides,
                "padding": padding,
                "data_format": data_format,
                "dilation_rate": dilation_rate,
                "groups": groups,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )
