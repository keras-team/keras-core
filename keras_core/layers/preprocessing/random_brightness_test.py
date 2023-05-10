import numpy as np

from keras_core import layers
from keras_core import testing


class RandomBrightnessTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            layers.RandomBrightness,
            init_kwargs={
                "factor": 0.75,
                "value_range": (20, 200),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_brightness_inference(self):
        seed = 3481
        layer = layers.RandomBrightness([0, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_output(self):
        seed = 2390

        # Always scale up, but randomly between 0 ~ 255
        layer = layers.RandomBrightness([0, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        diff = output - inputs
        self.assertTrue(np.amin(diff) >= 0)
        self.assertTrue(np.mean(diff) > 0)

        # Always scale down, but randomly between 0 ~ 255
        layer = layers.RandomBrightness([-1.0, 0.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        diff = output - inputs
        self.assertTrue(np.amax(diff) <= 0)
        self.assertTrue(np.mean(diff) < 0)