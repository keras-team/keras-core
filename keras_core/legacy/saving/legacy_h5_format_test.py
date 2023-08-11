import os

import numpy as np
import pytest
import tensorflow as tf

import keras_core
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core import backend
from keras_core.legacy.saving import legacy_h5_format
from keras_core.legacy.saving import serialization
from keras_core.saving import object_registration

# TODO: more thorough testing. Correctness depends
# on exact weight ordering for each layer, so we need
# to test across all types of layers.


def get_sequential_model(keras):
    return keras.Sequential(
        [
            keras.layers.Input((3,), batch_size=2),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.BatchNormalization(
                moving_mean_initializer="uniform", gamma_initializer="uniform"
            ),
            keras.layers.Dense(5, activation="softmax"),
        ]
    )


def get_functional_model(keras):
    inputs = keras.Input((3,), batch_size=2)
    x = keras.layers.Dense(4, activation="relu")(inputs)
    residual = x
    x = keras.layers.BatchNormalization(
        moving_mean_initializer="uniform", gamma_initializer="uniform"
    )(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.add([x, residual])
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_subclassed_model(keras):
    class MyModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense_1 = keras.layers.Dense(3, activation="relu")
            self.dense_2 = keras.layers.Dense(1, activation="sigmoid")

        def call(self, x):
            return self.dense_2(self.dense_1(x))

    model = MyModel()
    model(np.random.random((2, 3)))
    return model


@pytest.mark.requires_trainable_backend
class LegacyH5WeightsTest(testing.TestCase):
    def _check_reloading_weights(self, ref_input, model, tf_keras_model):
        ref_output = tf_keras_model(ref_input)
        initial_weights = model.get_weights()
        # Check weights only file
        temp_filepath = os.path.join(self.get_temp_dir(), "weights.h5")
        tf_keras_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)
        model.set_weights(initial_weights)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)

    def test_sequential_model_weights(self):
        model = get_sequential_model(keras_core)
        tf_keras_model = get_sequential_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def test_functional_model_weights(self):
        model = get_functional_model(keras_core)
        tf_keras_model = get_functional_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def test_subclassed_model_weights(self):
        model = get_subclassed_model(keras_core)
        tf_keras_model = get_subclassed_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)


@pytest.mark.requires_trainable_backend
class LegacyH5WholeModelTest(testing.TestCase):
    def _check_reloading_model(self, ref_input, model):
        # Whole model file
        ref_output = model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)

    def test_sequential_model(self):
        model = get_sequential_model(keras_core)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def test_functional_model(self):
        model = get_functional_model(keras_core)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def test_compiled_model_with_various_layers(self):
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))

        model.compile(optimizer="rmsprop", loss="mse")
        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model)

    def test_saving_lambda(self):
        mean = np.random.random((4, 2, 3))
        std = np.abs(np.random.random((4, 2, 3))) + 1e-5
        inputs = layers.Input(shape=(4, 2, 3))
        output = layers.Lambda(
            lambda image, mu, std: (image - mu) / std,
            arguments={"mu": mean, "std": std},
        )(inputs)
        model = models.Model(inputs, output)
        model.compile(loss="mse", optimizer="sgd", metrics=["acc"])

        temp_filepath = os.path.join(self.get_temp_dir(), "lambda_model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)

        self.assertAllClose(mean, loaded.layers[1].arguments["mu"])
        self.assertAllClose(std, loaded.layers[1].arguments["std"])

    def test_saving_include_optimizer_false(self):
        model = models.Sequential()
        model.add(layers.Dense(1))
        model.compile("adam", loss="mse")
        x, y = np.ones((10, 10)), np.ones((10, 1))
        model.fit(x, y)
        ref_output = model(x)

        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath, include_optimizer=False)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(x)

        # Assert that optimizer does not exist in loaded model
        with self.assertRaises(AttributeError):
            _ = loaded.optimizer

        # Compare output
        self.assertAllClose(ref_output, output, atol=1e-5)

    def test_shared_objects(self):
        class OuterLayer(layers.Layer):
            def __init__(self, inner_layer):
                super().__init__()
                self.inner_layer = inner_layer

            def call(self, inputs):
                return self.inner_layer(inputs)

            def get_config(self):
                return {
                    "inner_layer": serialization.serialize_keras_object(
                        self.inner_layer
                    )
                }

            @classmethod
            def from_config(cls, config):
                return cls(
                    serialization.deserialize_keras_object(
                        config["inner_layer"]
                    )
                )

        class InnerLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.v = self.add_weight(name="v", shape=[], initializer="zeros", dtype=tf.float32)

            def call(self, inputs):
                return self.v + inputs

            @classmethod
            def from_config(cls, config):
                return cls()

        # Create a model with 2 output layers that share the same inner layer.
        inner_layer = InnerLayer()
        outer_layer_1 = OuterLayer(inner_layer)
        outer_layer_2 = OuterLayer(inner_layer)
        input_ = layers.Input(shape=(1,))
        model = models.Model(
            inputs=input_,
            outputs=[outer_layer_1(input_), outer_layer_2(input_)],
        )

        one = tf.convert_to_tensor([1])
        # Changes to the shared layer should affect both outputs.
        model.layers[1].inner_layer.v.assign(5)
        self.assertAllEqual(model(one), [6.0, 6.0])
        model.layers[1].inner_layer.v.assign(3)
        self.assertAllEqual(model(one), [4.0, 4.0])

        # After loading, changes to the shared layer should still affect both
        # outputs.
        def _do_assertions(loaded):
            loaded.layers[1].inner_layer.v.assign(5)
            self.assertAllEqual(loaded(one), [6.0, 6.0])
            loaded.layers[1].inner_layer.v.assign(3)
            self.assertAllEqual(loaded(one), [4.0, 4.0])
            loaded.layers[2].inner_layer.v.assign(5)
            self.assertAllEqual(loaded(one), [6.0, 6.0])
            loaded.layers[2].inner_layer.v.assign(3)
            self.assertAllEqual(loaded(one), [4.0, 4.0])

        # We'd like to make sure we only attach shared object IDs when strictly
        # necessary, so we'll recursively traverse the generated config to count
        # whether we have the exact number we expect.
        def _get_all_keys_recursive(dict_or_iterable):
            if isinstance(dict_or_iterable, dict):
                for key in dict_or_iterable.keys():
                    yield key
                for key in _get_all_keys_recursive(dict_or_iterable.values()):
                    yield key
            elif isinstance(dict_or_iterable, str):
                return
            else:
                try:
                    for item in dict_or_iterable:
                        for key in _get_all_keys_recursive(item):
                            yield key
                # Not an iterable or dictionary
                except TypeError:
                    return

        with object_registration.CustomObjectScope(
            {"OuterLayer": OuterLayer, "InnerLayer": InnerLayer}
        ):
            # Test saving and loading to disk
            temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
            legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
            loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
            _do_assertions(loaded)

            # Test recreating directly from config
            config = model.get_config()
            key_count = collections.Counter(_get_all_keys_recursive(config))
            self.assertEqual(key_count[serialization.SHARED_OBJECT_KEY], 2)
            loaded = keras.Model.from_config(config)
            _do_assertions(loaded)


    # def test_rnn_model(self):
    #     inputs = layers.Input([10, 91], name="train_input")
    #     rnn_layers = [
    #         layers.LSTMCell(
    #             size, recurrent_dropout=0, name="rnn_cell%d" % i
    #         )
    #         for i, size in enumerate([512, 512])
    #     ]
    #     rnn_output = layers.RNN(
    #         rnn_layers, return_sequences=True, name="rnn_layer"
    #     )(inputs)
    #     pred_feat = layers.Dense(91, name="prediction_features")(
    #         rnn_output
    #     )
    #     pred = layers.Softmax()(pred_feat)
    #     model = models.Model(inputs=[inputs], outputs=[pred, pred_feat])


@pytest.mark.requires_trainable_backend
class LegacyH5BackwardsCompatTest(testing.TestCase):
    def _check_reloading_model(self, ref_input, model, tf_keras_model):
        # Whole model file
        ref_output = tf_keras_model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        tf_keras_model.save(temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)

    def test_sequential_model(self):
        model = get_sequential_model(keras_core)
        tf_keras_model = get_sequential_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def test_functional_model(self):
        tf_keras_model = get_functional_model(tf.keras)
        model = get_functional_model(keras_core)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def test_compiled_model_with_various_layers(self):
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))
        model.compile(optimizer="rmsprop", loss="mse")

        tf_keras_model = tf.keras.Sequential()
        tf_keras_model.add(tf.keras.layers.Dense(2, input_shape=(3,)))
        tf_keras_model.add(tf.keras.layers.RepeatVector(3))
        tf_keras_model.add(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3))
        )
        tf_keras_model.compile(optimizer="rmsprop", loss="mse")

        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)
