"""Tests for inference-only model/layer exporting utilities."""
import os

import numpy as np
import tensorflow as tf
import pytest
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core import layers
from keras_core import models
from keras_core import utils
from keras_core.export import export_lib


def get_model():
    layer_list = [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    model = models.Sequential(layer_list)
    return model


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Export only currently supports the TF backend.",
)
class ExportArchiveTest(testing.TestCase, parameterized.TestCase):
    def test_standard_model_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.serve(ref_input).numpy(), atol=1e-6
        )

    def test_low_level_model_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        # Test variable tracking
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        self.assertLen(export_archive.variables, 8)
        self.assertLen(export_archive.trainable_variables, 6)
        self.assertLen(export_archive.non_trainable_variables, 2)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        # Test registering an endpoint that is a tf.function (called)
        my_endpoint(ref_input)  # Trace fn

        export_archive.add_endpoint(
            "call",
            my_endpoint,
        )
        export_archive.write_out(temp_filepath)

        revived_model = tf.saved_model.load(temp_filepath)
        self.assertFalse(hasattr(revived_model, "_tracked"))
        self.assertAllClose(
            ref_output, revived_model.call(ref_input).numpy(), atol=1e-6
        )
        self.assertLen(revived_model.variables, 8)
        self.assertLen(revived_model.trainable_variables, 6)
        self.assertLen(revived_model.non_trainable_variables, 2)

        # Test registering an endpoint that is NOT a tf.function
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 10),
                    dtype=tf.float32,
                )
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.call(ref_input).numpy(), atol=1e-6
        )

    def test_layer_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.BatchNormalization()
        ref_input = tf.random.normal((3, 10))
        ref_output = layer(ref_input).numpy()  # Build layer (important)

        export_archive = export_lib.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 10),
                    dtype=tf.float32,
                )
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_layer.call(ref_input).numpy(), atol=1e-6
        )

    def test_multi_input_output_functional_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        x1 = layers.Input((2,))
        x2 = layers.Input((2,))
        y1 = layers.Dense(3)(x1)
        y2 = layers.Dense(3)(x2)
        model = models.Model([x1, x2], [y1, y2])

        ref_inputs = [tf.random.normal((3, 2)), tf.random.normal((3, 2))]
        ref_outputs = model(ref_inputs)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "serve",
            model.call,
            input_signature=[
                [
                    tf.TensorSpec(
                        shape=(None, 2),
                        dtype=tf.float32,
                    ),
                    tf.TensorSpec(
                        shape=(None, 2),
                        dtype=tf.float32,
                    ),
                ]
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_outputs[0].numpy(),
            revived_model.serve(ref_inputs)[0].numpy(),
            atol=1e-6,
        )
        self.assertAllClose(
            ref_outputs[1].numpy(),
            revived_model.serve(ref_inputs)[1].numpy(),
            atol=1e-6,
        )

        # Now test dict inputs
        model = models.Model({"x1": x1, "x2": x2}, [y1, y2])

        ref_inputs = {
            "x1": tf.random.normal((3, 2)),
            "x2": tf.random.normal((3, 2)),
        }
        ref_outputs = model(ref_inputs)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "serve",
            model.call,
            input_signature=[
                {
                    "x1": tf.TensorSpec(
                        shape=(None, 2),
                        dtype=tf.float32,
                    ),
                    "x2": tf.TensorSpec(
                        shape=(None, 2),
                        dtype=tf.float32,
                    ),
                }
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_outputs[0].numpy(),
            revived_model.serve(ref_inputs)[0].numpy(),
            atol=1e-6,
        )
        self.assertAllClose(
            ref_outputs[1].numpy(),
            revived_model.serve(ref_inputs)[1].numpy(),
            atol=1e-6,
        )

    def test_model_with_lookup_table(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        text_vectorization = layers.TextVectorization()
        text_vectorization.adapt(["one two", "three four", "five six"])
        model = models.Sequential(
            [
                text_vectorization,
                layers.Embedding(10, 32),
                layers.Dense(1),
            ]
        )
        ref_input = tf.convert_to_tensor(["one two three four"])
        ref_output = model(ref_input).numpy()

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.serve(ref_input).numpy(), atol=1e-6
        )

    def test_track_multiple_layers(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        layer_1 = layers.Dense(2)
        ref_input_1 = tf.random.normal((3, 4))
        ref_output_1 = layer_1(ref_input_1).numpy()
        layer_2 = layers.Dense(3)
        ref_input_2 = tf.random.normal((3, 5))
        ref_output_2 = layer_2(ref_input_2).numpy()

        export_archive = export_lib.ExportArchive()
        export_archive.add_endpoint(
            "call_1",
            layer_1.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 4),
                    dtype=tf.float32,
                ),
            ],
        )
        export_archive.add_endpoint(
            "call_2",
            layer_2.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 5),
                    dtype=tf.float32,
                ),
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output_1,
            revived_layer.call_1(ref_input_1).numpy(),
            atol=1e-6,
        )
        self.assertAllClose(
            ref_output_2,
            revived_layer.call_2(ref_input_2).numpy(),
            atol=1e-6,
        )

    def test_non_standard_layer_signature(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = layers.MultiHeadAttention(2, 2)
        x1 = tf.random.normal((3, 2, 2))
        x2 = tf.random.normal((3, 2, 2))
        ref_output = layer(x1, x2).numpy()  # Build layer (important)
        export_archive = export_lib.ExportArchive()
        export_archive.track(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 2, 2),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(None, 2, 2),
                    dtype=tf.float32,
                ),
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output,
            revived_layer.call(query=x1, value=x2).numpy(),
            atol=1e-6,
        )

    def test_variable_collection(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(2),
                layers.Dense(2),
            ]
        )

        # Test variable tracking
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 10),
                    dtype=tf.float32,
                )
            ],
        )
        export_archive.add_variable_collection(
            "my_vars", model.layers[1].weights
        )
        self.assertLen(export_archive.my_vars, 2)
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertLen(revived_model.my_vars, 2)

    def test_export_model_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Model has not been built
        model = models.Sequential([layers.Dense(2)])
        with self.assertRaisesRegex(ValueError, "It must be built"):
            export_lib.export_model(model, temp_filepath)

        # Subclassed model has not been called
        class MyModel(models.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense = layers.Dense(2)

            def build(self, input_shape):
                self.dense.build(input_shape)
                self.built = True

            def call(self, x):
                return self.dense(x)

        model = MyModel()
        model.build((2, 3))
        with self.assertRaisesRegex(ValueError, "It must be called"):
            export_lib.export_model(model, temp_filepath)

    def test_export_archive_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Dense(2)])
        model(tf.random.normal((2, 3)))

        # Endpoint name reuse
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            "call",
            model.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 3),
                    dtype=tf.float32,
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "already taken"):
            export_archive.add_endpoint(
                "call",
                model.call,
                input_signature=[
                    tf.TensorSpec(
                        shape=(None, 3),
                        dtype=tf.float32,
                    )
                ],
            )

        # Write out with no endpoints
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(ValueError, "No endpoints have been set"):
            export_archive.write_out(temp_filepath)

        # Invalid object type
        with self.assertRaisesRegex(ValueError, "Invalid resource type"):
            export_archive = export_lib.ExportArchive()
            export_archive.track("model")

        # Set endpoint with no input signature
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(
            ValueError, "you must provide an `input_signature`"
        ):
            export_archive.add_endpoint(
                "call",
                model.call,
            )

        # Set endpoint that has never been called
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        with self.assertRaisesRegex(
            ValueError, "you must either provide a function"
        ):
            export_archive.add_endpoint(
                "call",
                my_endpoint,
            )

    def test_export_no_assets(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Case where there are legitimately no assets.
        model = models.Sequential([layers.Flatten()])
        model(tf.random.normal((2, 3)))
        export_archive = export_lib.ExportArchive()
        export_archive.add_endpoint(
            "call",
            model.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 3),
                    dtype=tf.float32,
                )
            ],
        )
        export_archive.write_out(temp_filepath)

    def test_model_export_method(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        model.export(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.serve(ref_input).numpy(), atol=1e-6
        )


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Export only currently supports the TF backend.",
)
class TestReloadedLayer(tf.test.TestCase, parameterized.TestCase):
    def test_reloading_export_archive(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        export_lib.export_model(model, temp_filepath)
        reloaded_layer = export_lib.ReloadedLayer(temp_filepath)
        self.assertAllClose(
            reloaded_layer(ref_input).numpy(), ref_output, atol=1e-7
        )
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )

        # Test fine-tuning
        new_model = models.Sequential([reloaded_layer])
        new_model.compile(optimizer="rmsprop", loss="mse")
        x = tf.random.normal((32, 10))
        y = tf.random.normal((32, 1))
        new_model.train_on_batch(x, y)
        new_output = reloaded_layer(ref_input).numpy()
        self.assertNotAllClose(new_output, ref_output, atol=1e-5)

        # Test that trainable can be set to False
        reloaded_layer.trainable = False
        new_model.compile(optimizer="rmsprop", loss="mse")
        x = tf.random.normal((32, 10))
        y = tf.random.normal((32, 1))
        new_model.train_on_batch(x, y)
        # The output must not have changed
        self.assertAllClose(
            reloaded_layer(ref_input).numpy(), new_output, atol=1e-7
        )

    def test_reloading_default_saved_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        tf.saved_model.save(model, temp_filepath)
        reloaded_layer = export_lib.ReloadedLayer(
            temp_filepath, call_endpoint="serving_default"
        )
        # The output is a dict, due to the nature of SavedModel saving.
        new_output = reloaded_layer(ref_input)
        self.assertAllClose(
            new_output[list(new_output.keys())[0]].numpy(),
            ref_output,
            atol=1e-7,
        )
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )

    def test_call_training(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        utils.set_random_seed(1337)
        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(10),
                layers.Dropout(0.99999),
            ]
        )
        export_archive = export_lib.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="call_inference",
            fn=lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.add_endpoint(
            name="call_training",
            fn=lambda x: model(x, training=True),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        reloaded_layer = export_lib.ReloadedLayer(
            temp_filepath,
            call_endpoint="call_inference",
            call_training_endpoint="call_training",
        )
        inference_output = reloaded_layer(
            tf.random.normal((1, 10)), training=False
        )
        training_output = reloaded_layer(
            tf.random.normal((1, 10)), training=True
        )
        self.assertAllClose(np.mean(training_output), 0.0, atol=1e-7)
        self.assertNotAllClose(np.mean(inference_output), 0.0, atol=1e-7)

    def test_serialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        export_lib.export_model(model, temp_filepath)
        reloaded_layer = export_lib.ReloadedLayer(temp_filepath)

        # Test reinstantiation from config
        config = reloaded_layer.get_config()
        rereloaded_layer = export_lib.ReloadedLayer.from_config(config)
        self.assertAllClose(
            rereloaded_layer(ref_input).numpy(), ref_output, atol=1e-7
        )

        # Test whole model saving with reloaded layer inside
        model = models.Sequential([reloaded_layer])
        temp_model_filepath = os.path.join(self.get_temp_dir(), "m.keras")
        model.save(temp_model_filepath, save_format="keras_v3")
        reloaded_model = models.Models.load_model(
            temp_model_filepath,
            custom_objects={"ReloadedLayer": export_lib.ReloadedLayer},
        )
        self.assertAllClose(
            reloaded_model(ref_input).numpy(), ref_output, atol=1e-7
        )

    def test_errors(self):
        # Test missing call endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Input((2,)), layers.Dense(3)])
        export_lib.export_model(model, temp_filepath)
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            export_lib.ReloadedLayer(temp_filepath, call_endpoint="wrong")

        # Test missing call training endpoint
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            export_lib.ReloadedLayer(
                temp_filepath,
                call_endpoint="serve",
                call_training_endpoint="wrong",
            )
