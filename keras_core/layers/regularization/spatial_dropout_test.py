import numpy as np

from keras_core import layers
from keras_core.testing import test_case


class SpatialDropoutTest(test_case.TestCase):
    def test_spatial_dropout_1d(self):
        self.run_layer_test(
            layers.SpatialDropout1D,
            init_kwargs={"rate": 0.5},
            call_kwargs={"training": True},
            input_shape=(2, 3, 4),
        )

        self.run_layer_test(
            layers.SpatialDropout1D,
            init_kwargs={"rate": 0.5},
            call_kwargs={"training": False},
            input_shape=(2, 3, 4),
        )

    def test_spatial_dropout_2d(self):
        self.run_layer_test(
            layers.SpatialDropout2D,
            init_kwargs={"rate": 0.5},
            call_kwargs={"training": True},
            input_shape=(2, 3, 4, 5),
        )

        self.run_layer_test(
            layers.SpatialDropout2D,
            init_kwargs={"rate": 0.5, "data_format": "channels_first"},
            call_kwargs={"training": True},
            input_shape=(2, 3, 4, 5),
        )

    def test_spatial_dropout_3d(self):
        self.run_layer_test(
            layers.SpatialDropout3D,
            init_kwargs={"rate": 0.5},
            call_kwargs={"training": True},
            input_shape=(2, 3, 4, 4, 5),
        )

        self.run_layer_test(
            layers.SpatialDropout3D,
            init_kwargs={"rate": 0.5, "data_format": "channels_first"},
            call_kwargs={"training": True},
            input_shape=(2, 3, 4, 4, 5),
        )

    def test_spatial_dropout_1D_dynamic(self):
        inputs = layers.Input((3, 2))
        layer = layers.SpatialDropout1D(0.5)
        layer(inputs, training=True)

    def test_spatial_dropout_1D_correctness(self):
        inputs = np.ones((10, 3, 10))
        layer = layers.SpatialDropout1D(0.5)
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, :], outputs[:, 1, :])

    def test_spatial_dropout_2D_dynamic(self):
        inputs = layers.Input((3, 2, 4))
        layer = layers.SpatialDropout2D(0.5)
        layer(inputs, training=True)

    def test_spatial_dropout_2D_correctness(self):
        inputs = np.ones((10, 3, 3, 10))
        layer = layers.SpatialDropout2D(0.5)
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, 0, :], outputs[:, 1, 1, :])

    def test_spatial_dropout_3D_dynamic(self):
        inputs = layers.Input((3, 2, 4, 2))
        layer = layers.SpatialDropout3D(0.5)
        layer(inputs, training=True)

    def test_spatial_dropout_3D_correctness(self):
        inputs = np.ones((10, 3, 3, 3, 10))
        layer = layers.SpatialDropout3D(0.5)
        outputs = layer(inputs, training=True)
        self.assertAllClose(outputs[:, 0, 0, 0, :], outputs[:, 1, 1, 1, :])
