import numpy as np
import pytest
from absl.testing import parameterized

from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.models.cloning import clone_model


def get_functional_model(shared_layers=False):
    inputs = layers.Input(shape=(3,))
    x = layers.Dense(2)(inputs)
    if shared_layers:
        layer = layers.Dense(2, name="shared")
        x = layer(x)
        x = layer(x)
    outputs = layers.Dense(2)(x)
    model = models.Model(inputs, outputs)
    return model


def get_sequential_model(explicit_input=True):
    model = models.Sequential()
    if explicit_input:
        model.add(layers.Input(shape=(3,)))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))
    return model


def get_subclassed_model():
    class ExampleModel(models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.d1 = layers.Dense(2)
            self.d2 = layers.Dense(2)

        def call(self, x):
            return self.d2(self.d1(x))

    return ExampleModel()


@pytest.mark.requires_trainable_backend
class CloneModelTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("functional", get_functional_model),
        ("sequential", get_sequential_model),
        (
            "deferred_sequential",
            lambda: get_sequential_model(explicit_input=False),
        ),
        ("subclassed", get_subclassed_model),
    )
    def test_cloning_correctness(self, model_fn):
        ref_input = np.random.random((2, 3))
        model = model_fn()
        new_model = clone_model(model)
        ref_output = model(ref_input)
        new_model(ref_input)  # Maybe needed to build the model
        new_model.set_weights(model.get_weights())
        output = new_model(ref_input)
        self.assertAllClose(ref_output, output)

    @parameterized.named_parameters(
        ("functional", get_functional_model),
        ("sequential", get_sequential_model),
    )
    def test_custom_clone_function(self, model_fn):
        def clone_function(layer):
            config = layer.get_config()
            config["name"] = config["name"] + "_custom"
            return layer.__class__.from_config(config)

        model = model_fn()
        new_model = clone_model(model, clone_function=clone_function)
        for l1, l2 in zip(model.layers, new_model.layers):
            if not isinstance(l1, layers.InputLayer):
                self.assertEqual(l2.name, l1.name + "_custom")

    def test_shared_layers_cloning(self):
        model = get_functional_model(shared_layers=True)
        new_model = clone_model(model)
        self.assertLen(new_model.layers, 4)

    def test_cloning_with_dummy_model_raises_error(self):
        class DummyModel:
            pass

        dummy_model = DummyModel()
        with pytest.raises(
            ValueError,
            match="Arguments clone_function and input_tensors are only supported",
        ):
            clone_model(dummy_model)

    def test_clone_sequential_with_non_sequential_raises_error(self):
        non_sequential_model = get_functional_model()
        with pytest.raises(
            ValueError,
            match="Expected `model` argument to be a `Sequential` model instance",
        ):
            _clone_sequential_model(non_sequential_model)

    def test_clone_sequential_with_non_callable_clone_function_raises_error(
        self,
    ):
        sequential_model = get_sequential_model()
        with pytest.raises(
            ValueError,
            match="Expected `clone_function` argument to be a callable",
        ):
            _clone_sequential_model(
                sequential_model, clone_function="not_callable"
            )

    def test_clone_sequential_with_incorrect_input_tensors_format_raises_error(
        self,
    ):
        sequential_model = get_sequential_model()
        with pytest.raises(
            ValueError,
            match="Argument `input_tensors` must contain a single tensor",
        ):
            _clone_sequential_model(
                sequential_model, input_tensors=[input_tensor, input_tensor]
            )

    def test_clone_functional_with_non_functional_raises_error(self):
        non_functional_model = get_sequential_model()
        with pytest.raises(
            ValueError,
            match="Expected `model` argument to be a Functional Model instance",
        ):
            _clone_functional_model(non_functional_model)

    def test_clone_functional_with_non_callable_clone_function_raises_error(
        self,
    ):
        functional_model = get_functional_model()
        with pytest.raises(
            ValueError,
            match="Expected `clone_function` argument to be a callable",
        ):
            _clone_functional_model(
                functional_model, clone_function="not_callable"
            )

    def test_clone_functional_with_non_keras_tensor_raises_error(self):
        functional_model = get_functional_model()
        non_keras_tensor = "not_a_keras_tensor"
        with pytest.raises(
            ValueError,
            match="All entries in `input_tensors` must be KerasTensors",
        ):
            _clone_functional_model(
                functional_model, input_tensors=non_keras_tensor
            )
