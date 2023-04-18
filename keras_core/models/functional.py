import warnings

from tensorflow import nest

from keras_core import backend
from keras_core import operations as ops
from keras_core.layers.layer import Layer
from keras_core.models.model import Model
from keras_core.operations.function import Function
from keras_core.utils import tracking


class Functional(Function, Model):
    """
    Add support for extra call arguments compared to Function:
    training, masks

    Add support for arg standardization:
    - list/dict duality
    - upranking

    Override .layers

    Symbolic add_loss
    """

    @tracking.no_automatic_dependency_tracking
    def __init__(self, inputs, outputs, name=None, **kwargs):
        # This is used by the Model class, since we have some logic to swap the
        # class in the __new__ method, which will lead to __init__ get invoked
        # twice. Using the skip_init to skip one of the invocation of __init__
        # to avoid any side effects
        skip_init = kwargs.pop("skip_init", False)
        if skip_init:
            return
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if not isinstance(v, backend.KerasTensor):
                    raise ValueError(
                        "When providing `inputs` as a dict, all values in the dict "
                        f"must be KerasTensors. Received: inputs={inputs} including "
                        f"invalid value {v} of type {type(v)}"
                    )
                if k != v.name:
                    # TODO: maybe make this a warning
                    raise ValueError(
                        "When providing `inputs` as a dict, all keys in the dict "
                        "must match the names of the corresponding tensors. "
                        f"Received key '{k}' mapping to value {v} which has name '{v.name}'. "
                        f"Change the tensor name to '{k}' (via `Input(..., name='{k}')`)"
                    )
        elif isinstance(inputs, (list, tuple)):
            for x in inputs:
                if not isinstance(x, backend.KerasTensor):
                    raise ValueError(
                        "When providing `inputs` as a list/tuple, all values in the list/tuple "
                        f"must be KerasTensors. Received: inputs={inputs} including "
                        f"invalid value {x} of type {type(x)}"
                    )
        elif not isinstance(inputs, backend.KerasTensor):
            raise ValueError(
                f"Unrecognized type for `inputs`: {inputs} (of type {type(inputs)})"
            )
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if not isinstance(v, backend.KerasTensor):
                    raise ValueError(
                        "When providing `outputs` as a dict, all values in the dict "
                        f"must be KerasTensors. Received: outputs={outputs} including "
                        f"invalid value {v} of type {type(v)}"
                    )
        elif isinstance(outputs, (list, tuple)):
            for x in outputs:
                if not isinstance(x, backend.KerasTensor):
                    raise ValueError(
                        "When providing `outputs` as a list/tuple, all values in the list/tuple "
                        f"must be KerasTensors. Received: outputs={outputs} including "
                        f"invalid value {x} of type {type(x)}"
                    )
        elif not isinstance(outputs, backend.KerasTensor):
            raise ValueError(
                f"Unrecognized type for `outputs`: {outputs} (of type {type(outputs)})"
            )

        super().__init__(inputs, outputs, name=name, **kwargs)
        self._layers = self.layers
        self.built = True

    @property
    def layers(self):
        layers = []
        for operation in self._operations:
            if isinstance(operation, Layer):
                layers.append(operation)
        return layers

    def call(self, inputs, training=None, mask=None):
        # Add support for traning, masking
        inputs = self._standardize_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = self._flatten_to_reference_inputs(mask)
            for x, mask in zip(inputs, masks):
                x._keras_mask = mask
        return self._run_through_graph(
            inputs, operation_fn=lambda op: operation_fn(op, training=training)
        )

    def compute_output_spec(self, inputs, training=None, mask=None):
        # From Function
        return super().compute_output_spec(inputs)

    def _assert_input_compatibility(self, *args):
        return super(Model, self)._assert_input_compatibility(*args)

    def _flatten_to_reference_inputs(self, inputs, allow_extra_keys=True):
        if isinstance(inputs, dict):
            ref_inputs = self._inputs_struct
            if not nest.is_nested(ref_inputs):
                ref_inputs = [self._nested_inputs]
            if isinstance(ref_inputs, dict):
                # In the case that the graph is constructed with dict input
                # tensors, We will use the original dict key to map with the
                # keys in the input data. Note that the model.inputs is using
                # nest.flatten to process the input tensors, which means the
                # dict input tensors are ordered by their keys.
                ref_input_names = sorted(ref_inputs.keys())
            else:
                ref_input_names = [
                    inp._keras_history.operation.name for inp in ref_inputs
                ]
            # Raise an warning if there are more input data comparing to input
            # tensor
            if allow_extra_keys and len(inputs) > len(ref_input_names):
                warnings.warn(
                    "Input dict contained keys {} which did not match any "
                    "model input. They will be ignored by the model.".format(
                        [n for n in inputs.keys() if n not in ref_input_names]
                    ),
                    stacklevel=2,
                )
            # Flatten in the order `Input`s were passed during Model
            # construction.
            return [inputs[n] for n in ref_input_names]
        # Otherwise both ref inputs and inputs will already be in same order.
        return nest.flatten(inputs)

    def _adjust_input_rank(self, flat_inputs):
        flat_ref_shapes = [x.shape for x in self._inputs]
        names = [x.name for x in self._inputs]
        adjusted = []
        for x, ref_shape in zip(flat_inputs, flat_ref_shapes):
            x_rank = len(x.shape)
            ref_rank = len(ref_shape)
            if x_rank == ref_rank:
                adjusted.append(x)
                continue
            if x_rank == ref_rank + 1:
                if x.shape[-1] == 1:
                    adjusted.append(ops.squeeze(x, axis=-1))
                    continue
            if x_rank == ref_rank - 1:
                if ref_shape[-1] == 1:
                    adjusted.append(ops.expand_dims(x, axis=-1))
                    continue
            raise ValueError(
                f"Invalid input shape for input {x}. Expected shape "
                f"{ref_shape}, but input has incompatible shape {x.shape}"
            )
        # Add back metadata.
        for i in range(len(flat_inputs)):
            if hasattr(flat_inputs[i], "_keras_history"):
                adjusted[i]._keras_history = flat_inputs[i]._keras_history
            if hasattr(flat_inputs[i], "_keras_mask"):
                adjusted[i]._keras_mask = flat_inputs[i]._keras_mask
        return adjusted

    def _standardize_inputs(self, inputs):
        flat_inputs = self._flatten_to_reference_inputs(inputs)
        return self._adjust_input_rank(flat_inputs)

    def add_loss(self, loss):
        # Symbolic only. TODO
        raise NotImplementedError


def operation_fn(operation, training):
    def call(*args, **kwargs):
        if (
            hasattr(operation, "_call_has_training_arg")
            and operation._call_has_training_arg()
        ):
            kwargs["training"] = training
        return operation(*args, **kwargs)

    return call
