import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import testing


class UpSampling3dTest(testing.TestCase):
    def test_upsampling_3d(self):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 10
        input_len_dim2 = 11
        input_len_dim3 = 12

        for data_format in ["channels_first", "channels_last"]:
            if data_format == "channels_first":
                inputs = np.random.rand(
                    num_samples,
                    stack_size,
                    input_len_dim1,
                    input_len_dim2,
                    input_len_dim3,
                )
            else:
                inputs = np.random.rand(
                    num_samples,
                    input_len_dim1,
                    input_len_dim2,
                    input_len_dim3,
                    stack_size,
                )

            # basic test
            self.run_layer_test(
                layers.UpSampling3D,
                init_kwargs={"size": (2, 2, 2), "data_format": data_format},
                input_shape=inputs.shape,
            )

            for length_dim1 in [2, 3]:
                for length_dim2 in [2]:
                    for length_dim3 in [3]:
                        layer = layers.UpSampling3D(
                            size=(length_dim1, length_dim2, length_dim3),
                            data_format=data_format,
                        )
                        layer.build(inputs.shape)
                        np_output = layer(inputs=backend.Variable(inputs))
                        if data_format == "channels_first":
                            assert (
                                np_output.shape[2]
                                == length_dim1 * input_len_dim1
                            )
                            assert (
                                np_output.shape[3]
                                == length_dim2 * input_len_dim2
                            )
                            assert (
                                np_output.shape[4]
                                == length_dim3 * input_len_dim3
                            )
                        else:  # tf
                            assert (
                                np_output.shape[1]
                                == length_dim1 * input_len_dim1
                            )
                            assert (
                                np_output.shape[2]
                                == length_dim2 * input_len_dim2
                            )
                            assert (
                                np_output.shape[3]
                                == length_dim3 * input_len_dim3
                            )

                        # compare with numpy
                        if data_format == "channels_first":
                            expected_out = np.repeat(
                                inputs, length_dim1, axis=2
                            )
                            expected_out = np.repeat(
                                expected_out, length_dim2, axis=3
                            )
                            expected_out = np.repeat(
                                expected_out, length_dim3, axis=4
                            )
                        else:  # tf
                            expected_out = np.repeat(
                                inputs, length_dim1, axis=1
                            )
                            expected_out = np.repeat(
                                expected_out, length_dim2, axis=2
                            )
                            expected_out = np.repeat(
                                expected_out, length_dim3, axis=3
                            )

                        np.testing.assert_allclose(np_output, expected_out)

    def test_upsampling_3d_correctness(self):
        input_shape = (2, 1, 2, 1, 3)
        x = np.arange(np.prod(input_shape)).reshape(input_shape)
        np.testing.assert_array_equal(
            layers.UpSampling3D(size=(2, 2, 2))(x),
            np.array(
                [
                    [
                        [
                            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                            [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                            [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                        ],
                        [
                            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                            [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                            [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                        ],
                    ],
                    [
                        [
                            [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                            [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                            [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                            [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                        ],
                        [
                            [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                            [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                            [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                            [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                        ],
                    ],
                ]
            ),
        )
