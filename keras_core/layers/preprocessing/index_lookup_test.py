import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.saving import saving_api


@pytest.mark.skipif(
    backend.backend() == "numpy", reason="Failing for numpy backend."
)
class IndexLookupLayerTest(testing.TestCase, parameterized.TestCase):
    def test_basics_string_vocab(self):
        # Case: adapt + list inputs
        adapt_data = ["one", "one", "one", "two", "two", "three"]
        input_data = ["one", "two", "four"]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: numpy array input
        output = layer(np.array(input_data))
        self.assertEqual(list(output), [2, 3, 1])

        # Case: fixed vocab + list inputs
        vocabulary = ["one", "two", "three"]
        layer = layers.IndexLookup(vocabulary=vocabulary, **kwargs)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: fixed vocab with special tokens + list inputs
        vocabulary_with_special_tokens = ["", "[OOV]", "one", "two", "three"]
        layer = layers.IndexLookup(
            vocabulary=vocabulary_with_special_tokens, **kwargs
        )
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary (with special tokens)
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary_with_special_tokens)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_basics_integer_vocab(self):
        # Case: adapt + list inputs
        adapt_data = [1, 1, 1, 2, 2, 3]
        input_data = [1, 2, 4]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": 0,
            "oov_token": -1,
            "vocabulary_dtype": "int64",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: numpy array input
        output = layer(np.array(input_data))
        self.assertEqual(list(output), [2, 3, 1])

        # Case: fixed vocab + list inputs
        vocabulary = [1, 2, 3]
        layer = layers.IndexLookup(vocabulary=vocabulary, **kwargs)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: fixed vocab with special tokens + list inputs
        vocabulary_with_special_tokens = [0, -1, 1, 2, 3]
        layer = layers.IndexLookup(
            vocabulary=vocabulary_with_special_tokens, **kwargs
        )
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary (with special tokens)
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary_with_special_tokens)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_max_tokens_adapt(self):
        adapt_data = [1, 1, 1, 2, 2, 3]
        input_data = [1, 2, 3, 4]
        kwargs = {
            "max_tokens": 4,
            "num_oov_indices": 1,
            "mask_token": 0,
            "oov_token": -1,
            "vocabulary_dtype": "int64",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_pad_to_max_tokens(self):
        vocabulary = [1, 2]
        input_data = [1, 2]
        kwargs = {
            "max_tokens": 5,
            "num_oov_indices": 1,
            "mask_token": 0,
            "oov_token": -1,
            "vocabulary_dtype": "int64",
            "vocabulary": vocabulary,
            "pad_to_max_tokens": True,
            "output_mode": "multi_hot",
        }
        layer = layers.IndexLookup(**kwargs)
        output = layer(input_data)
        self.assertAllClose(output, [0, 1, 1, 0, 0])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_output_modes(self):
        vocabulary = ["one", "two", "three"]
        single_sample_input_data = ["one", "two", "four"]
        batch_input_data = [["one", "two", "four", "two"]]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocabulary,
        }

        # int
        kwargs["output_mode"] = "int"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [2, 3, 1])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[2, 3, 1, 3]])

        # multi-hot
        kwargs["output_mode"] = "multi_hot"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [1, 1, 1, 0])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[1, 1, 1, 0]])

        # one-hot
        kwargs["output_mode"] = "one_hot"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])

        # count
        kwargs["output_mode"] = "count"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [1, 1, 1, 0])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[1, 1, 2, 0]])

        # tf-idf
        kwargs["output_mode"] = "tf_idf"
        kwargs["idf_weights"] = np.array([0.1, 0.2, 0.3])
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [0.2, 0.1, 0.2, 0.0])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[0.2, 0.1, 0.4, 0.0]])

    def test_sparse_outputs(self):
        # TODO
        pass

    def test_adapt_tf_idf(self):
        # Case: unbatched data
        adapt_data = ["one", "one", "one", "two", "two", "three"]
        input_data = ["one", "two", "four"]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "output_mode": "tf_idf",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        output = layer(input_data)
        # Document counts for one, two, three = [3, 2, 1]
        idf_weights = np.log(1 + len(adapt_data) / (1 + np.array([3, 2, 1])))
        self.assertAllClose(layer.idf_weights[1:], idf_weights)
        self.assertAllClose(output, [1.1337324, 0.91629076, 1.0986123, 0.0])
        # Case: batched data
        adapt_data = [["one", "one"], ["one", "two"], ["two", "three"]]
        input_data = [["one", "two"], ["two", "four"]]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "output_mode": "tf_idf",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        # Document counts for one, two, three = [2, 2, 1]
        idf_weights = np.log(1 + len(adapt_data) / (1 + np.array([2, 2, 1])))
        self.assertAllClose(layer.idf_weights[1:], idf_weights)
        output = layer(input_data)
        self.assertAllClose(
            output,
            [
                [0.0, 0.6931472, 0.6931472, 0.0],
                [0.76752836, 0.0, 0.6931472, 0.0],
            ],
        )

    def test_invert(self):
        vocabulary = ["one", "two", "three"]
        single_sample_input_data = [2, 3, 1]
        batch_input_data = [[2, 3, 1, 3]]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocabulary,
            "invert": True,
            "output_mode": "int",
        }
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertEqual(
            [w.decode("utf-8") for w in output.numpy()], ["one", "two", "[OOV]"]
        )
        output = layer(batch_input_data)
        self.assertEqual(
            [w.decode("utf-8") for w in output.numpy()[0]],
            ["one", "two", "[OOV]", "two"],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires string input dtype"
    )
    def test_saving(self):
        # Test with adapt()
        vocabulary = ["one", "two", "three"]
        adapt_data = ["one", "one", "one", "two", "two", "three"]
        batch_input_data = np.array([["one", "two", "four"]])
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "output_mode": "int",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        model = models.Sequential(
            [
                layers.Input(shape=(None,), dtype="string"),
                layer,
            ]
        )
        output_1 = model(batch_input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        model = saving_api.load_model(path)
        output_2 = model(batch_input_data)
        self.assertAllClose(output_1, output_2)

        # Test when vocabulary is provided
        kwargs["vocabulary"] = vocabulary
        layer = layers.IndexLookup(**kwargs)
        model = models.Sequential(
            [
                layers.Input(shape=(None,), dtype="string"),
                layer,
            ]
        )
        output_1 = model(batch_input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        model = saving_api.load_model(path)
        output_2 = model(batch_input_data)
        self.assertAllClose(output_1, output_2)

    def test_adapt_with_tf_data(self):
        # Case: adapt + list inputs
        adapt_data = tf.data.Dataset.from_tensor_slices(
            ["one", "one", "one", "two", "two", "three"]
        ).batch(2)
        input_data = ["one", "two", "four"]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)
