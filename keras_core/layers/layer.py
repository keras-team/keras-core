"""Layer is an Operation with state.

Takes care of:

- Weights / variables (and tracking thereof)
- deferred build
- trainable argument value inference
- masking
- autocasting

And some more magic:

- add_loss
- metric tracking
- RNG seed tracking
- activity regularization
"""
import collections
import inspect
import warnings

import numpy as np
from tensorflow import nest

from keras_core import backend
from keras_core import initializers
from keras_core import mixed_precision
from keras_core import regularizers
from keras_core import utils
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend.common import global_state
from keras_core.layers import input_spec
from keras_core.metrics.metric import Metric
from keras_core.operations.operation import Operation
from keras_core.utils import python_utils
from keras_core.utils import summary_utils
from keras_core.utils import traceback_utils
from keras_core.utils.shape_utils import map_shape_structure
from keras_core.utils.tracking import Tracker

if backend.backend() == "tensorflow":
    from keras_core.backend.tensorflow.layer import TFLayer as BackendLayer
elif backend.backend() == "jax":
    from keras_core.backend.jax.layer import JaxLayer as BackendLayer
elif backend.backend() == "torch":
    from keras_core.backend.torch.layer import TorchLayer as BackendLayer
else:
    raise RuntimeError(
        f"Backend '{backend.backend()}' must implement a layer mixin class."
    )


@keras_core_export(["keras_core.Layer", "keras_core.layers.Layer"])
class Layer(BackendLayer, Operation):
    """This is the class from which all layers inherit.

    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables). State can be
    created:

    * in `__init__()`, for instancce via `self.add_weight()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the layer, and supplies the shape(s) of the input(s),
      which may not have been known at initialization time.

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a layer and then treat it as a callable.

    Args:
        trainable: Boolean, whether the layer's variables should be trainable.
        name: String name of the layer.
        dtype: The dtype of the layer's computations and weights. Can also be a
            `keras_core.mixed_precision.DTypePolicy`,
            which allows the computation and
            weight dtype to differ. Default of `None` means to use
            `keras_core.mixed_precision.dtype_policy()`,
            which is a `float32` policy unless set to different value
            (via `keras_core.mixed_precision.set_dtype_policy()`).

    Attributes:
        name: The name of the layer (string).
        dtype: The dtype of the layer's weights.
        variable_dtype: Dtype of the layer's variables.
        compute_dtype: The dtype of the layer's computations.
            Layers automatically cast inputs to this dtype, which causes
            the computations and output to also be in this dtype.
            When mixed precision is used with a
            `keras_core.mixed_precision.DTypePolicy`, this will be different
            than `variable_dtype`.
        trainable_weights: List of variables to be included in backprop.
        non_trainable_weights: List of variables that should not be
            included in backprop.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        trainable: Whether the layer should be trained (boolean), i.e.
            whether its potentially-trainable weights should be returned
            as part of `layer.trainable_weights`.
        input_spec: Optional (list of) `InputSpec` object(s) specifying the
            constraints on inputs that can be accepted by the layer.

    We recommend that descendants of `Layer` implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
        that do not depend on input shapes, using `add_weight()`,
        or other state.
    * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`, or other
        state. `__call__()` will automatically build the layer
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the layer to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
            2. `mask` (boolean tensor encoding masked timesteps in the input,
                used e.g. in RNN layers).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` and `mask` if the layer need them.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this layer. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the layer or a model that contains this layer.

    Examples:

    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).

    ```python
    class SimpleDense(Layer):
        def __init__(self, units=32):
            super().__init__()
            self.units = units

        # Create the state of the layer (weights)
        def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel",
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        # Defines the computation
        def call(self, inputs):
            return ops.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```

    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:

    ```python
    class ComputeSum(Layer):

      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="total",
          )

      def call(self, inputs):
          self.total.assign(self.total + ops.sum(inputs))
          return self.total

    my_sum = ComputeSum(2)
    x = ops.ones((2, 2))
    y = my_sum(x)

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```
    """

    def __init__(
        self,
        *,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs,
    ):
        BackendLayer.__init__(self)
        self._lock = False
        Operation.__init__(self, name=name)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        input_dim_arg = kwargs.pop("input_dim", None)
        if input_dim_arg is not None:
            input_shape_arg = (input_dim_arg,)
        else:
            input_shape_arg = kwargs.pop("input_shape", None)
        if input_shape_arg is not None:
            warnings.warn(
                "Do not pass an `input_shape`/`input_dim` argument to "
                "a layer. When using Sequential models, "
                "prefer using an `Input(shape)` object as the "
                "first layer in the model instead.",
                stacklevel=2,
            )
            self._input_shape_arg = input_shape_arg
        if kwargs:
            raise ValueError(
                "Unrecognized keyword arguments "
                f"passed to {self.__class__.__name__}: {kwargs}"
            )

        self.built = False
        self.dtype_policy = mixed_precision.resolve_policy(dtype)
        self.autocast = autocast
        self.input_spec = None
        self.supports_jit = True

        self._trainable = trainable
        self._layers = []
        self._metrics = []
        self._seed_generators = []
        self._losses = []
        self._trainable_variables = []
        self._non_trainable_variables = []
        self._supports_masking = not utils.is_default(self.compute_mask)
        # Whether to automatically convert (+ auto-cast) inputs to `call()`.
        self._convert_input_args = True
        # Whether to allow non-tensors as positional arguments in `call()`.
        self._allow_non_tensor_positional_args = False
        # Dict of shapes that were used to call `build()`.
        self._build_shapes_dict = None
        self._call_signature_parameters = [
            p.name for p in inspect.signature(self.call).parameters.values()
        ]

        self._tracker = Tracker(
            {
                "trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and x.trainable,
                    self._trainable_variables,
                ),
                "non_trainable_variables": (
                    lambda x: isinstance(x, backend.Variable)
                    and not x.trainable,
                    self._non_trainable_variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), self._metrics),
                "layers": (
                    lambda x: isinstance(x, Layer)
                    and not isinstance(x, Metric),
                    self._layers,
                ),
                "seed_generators": (
                    lambda x: isinstance(x, backend.random.SeedGenerator),
                    self._seed_generators,
                ),
            }
        )

    @utils.default
    def build(self, input_shape):
        self.built = True

    def get_build_config(self):
        """Returns a dictionary with the layer's input shape.

        This method returns a config dict that can be used by
        `build_from_config(config)` to create all states (e.g. Variables and
        Lookup tables) needed by the layer.

        By default, the config only contains the input shape that the layer
        was built with. If you're writing a custom layer that creates state in
        an unusual way, you should override this method to make sure this state
        is already created when Keras attempts to load its value upon model
        loading.

        Returns:
            A dict containing the input shape associated with the layer.
        """
        if self._build_shapes_dict is not None:
            if len(self._build_shapes_dict) == 1:
                return {
                    "input_shape": tuple(self._build_shapes_dict.values())[0],
                }
            else:
                return {"shapes_dict": self._build_shapes_dict}

    def build_from_config(self, config):
        """Builds the layer's states with the supplied config dict.

        By default, this method calls the `build(config["input_shape"])` method,
        which creates weights based on the layer's input shape in the supplied
        config. If your config contains other information needed to load the
        layer's state, you should override this method.

        Args:
            config: Dict containing the input shape associated with this layer.
        """
        if config:
            if "input_shape" in config:
                self.build(config["input_shape"])
                self._build_shapes_dict = config
            elif "shapes_dict" in config:
                self.build(**config["shapes_dict"])
                self._build_shapes_dict = config["shapes_dict"]
            self.built = True

    def add_variable(
        self,
        shape,
        initializer,
        dtype=None,
        trainable=True,
        regularizer=None,
        constraint=None,
        name=None,
    ):
        # TODO: handle layout
        self._check_super_called()
        initializer = initializers.get(initializer)
        variable = backend.Variable(
            initializer=initializer,
            shape=shape,
            dtype=dtype or self.variable_dtype,
            trainable=trainable,
            name=name,
        )
        # Will be added to layer.losses
        variable.regularizer = regularizer
        variable.constraint = constraint
        self._track_variable(variable)
        return variable

    def add_weight(self, *args, **kwargs):
        return self.add_variable(*args, **kwargs)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Sets trainable attribute for the layer and its sublayers.

        When this value is changed during training (e.g. with a
        `Callback`) you need to call the parent
        `Model.make_train_function` with `force=True` in order to
        recompile the training graph.

        Args:
            value: Boolean with the desired state for the layer's trainable
                attribute.
        """
        value = bool(value)
        self._trainable = value
        for v in self._trainable_variables:
            v.trainable = value
        for layer in self._layers:
            layer.trainable = value

    @property
    def variables(self):
        # Return only weights/rng state/metric variables
        # of all Layers, recursively.
        # Also deduplicate them.
        variables = []
        seen_ids = set()
        for v in self._trainable_variables + self._non_trainable_variables:
            if id(v) not in seen_ids:
                variables.append(v)
                seen_ids.add(id(v))
        for m in self._metrics:
            variables.extend(m.variables)
        for sg in self._seed_generators:
            variables.append(sg.state)
        for layer in self._layers:
            for v in layer.variables:
                if id(v) not in seen_ids:
                    variables.append(v)
                    seen_ids.add(id(v))
        return variables

    @property
    def trainable_variables(self):
        if not self.trainable:
            return []
        return [v for v in self.variables if v.trainable]

    @property
    def non_trainable_variables(self):
        if not self.trainable:
            return self.variables
        return [v for v in self.variables if not v.trainable]

    @property
    def weights(self):
        # Return only "own weights" of all Layers, recursively.
        # Also deduplicate them.
        weights = []
        seen_ids = set()
        for w in self._trainable_variables + self._non_trainable_variables:
            if id(w) not in seen_ids:
                weights.append(w)
                seen_ids.add(id(w))
        for layer in self._layers:
            for w in layer.weights:
                if id(w) not in seen_ids:
                    weights.append(w)
                    seen_ids.add(id(w))
        return weights

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        return [v for v in self.weights if v.trainable]

    @property
    def non_trainable_weights(self):
        if not self.trainable:
            return self.weights
        return [v for v in self.weights if not v.trainable]

    def get_weights(self):
        return [v.numpy() for v in self.weights]

    def set_weights(self, weights):
        layer_weights = self.weights
        if len(layer_weights) != len(weights):
            raise ValueError(
                f"You called `set_weights(weights)` on layer '{self.name}' "
                f"with a weight list of length {len(weights)}, but the layer "
                f"was expecting {len(layer_weights)} weights."
            )
        for variable, value in zip(layer_weights, weights):
            if variable.shape != value.shape:
                raise ValueError(
                    f"Layer {self.name} weight shape {variable.shape} "
                    "is not compatible with provided weight "
                    f"shape {value.shape}."
                )
            variable.assign(value)

    @property
    def dtype(self):
        """The dtype of the state (weights) of the layer."""
        return self.variable_dtype

    @property
    def compute_dtype(self):
        """The dtype of the computations performed by the layer."""
        return self.dtype_policy.compute_dtype

    @property
    def variable_dtype(self):
        """The dtype of the state (weights) of the layer."""
        return self.dtype_policy.variable_dtype

    @property
    def supports_masking(self):
        """Whether this layer supports computing a mask using `compute_mask`."""
        return self._supports_masking

    @supports_masking.setter
    def supports_masking(self, value):
        self._supports_masking = value

    @utils.default
    def compute_mask(self, inputs, previous_mask):
        return previous_mask

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        self._check_super_called()

        #####################################
        # 1. Convert any array arguments to tensors of correct dtype.
        def maybe_convert(x):
            if backend.is_tensor(x):
                if (
                    self.autocast
                    and backend.is_float_dtype(x.dtype)
                    and x.dtype != self.compute_dtype
                ):
                    return backend.cast(x, dtype=self.compute_dtype)
                return x
            elif isinstance(x, backend.KerasTensor):
                if (
                    self.autocast
                    and backend.is_float_dtype(x.dtype)
                    and x.dtype != self.compute_dtype
                ):
                    x.dtype = self.compute_dtype
                return x
            elif hasattr(x, "__array__"):
                return backend.convert_to_tensor(x, dtype=self.compute_dtype)
            return x

        if self._convert_input_args:
            args = nest.map_structure(maybe_convert, args)
            kwargs = nest.map_structure(maybe_convert, kwargs)

        ##########################################################
        # 2. Enforce that only tensors can be passed positionally.
        if not self._allow_non_tensor_positional_args:
            for arg in nest.flatten(args):
                if not isinstance(arg, KerasTensor) and not backend.is_tensor(
                    arg
                ):
                    raise ValueError(
                        "Only input tensors may be passed as "
                        "positional arguments. The following argument value "
                        f"should be passed as a keyword argument: {arg} "
                        f"(of type {type(arg)})"
                    )

        # Caches info about `call()` signature, args, kwargs.
        call_spec = CallSpec(self.call, args, kwargs)

        ############################################
        # 3. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(call_spec.first_arg)

        ################
        # 4. Call build.
        self._maybe_build(call_spec)

        ##########################
        # 5. Infer training value
        # Training phase for `Layer.call` is set via (in order of priority):
        # (1) The `training` argument passed to this `Layer.call`, if not None
        # (2) The training argument of an outer `Layer.call`.
        # (4) Any non-None default value for `training` in the call signature
        # (5) False (treating the layer as if it's in inference)

        # Maintains info about the `Layer.call` stack
        # across nested calls.
        call_context = self._get_call_context()

        # This is the value explicity passed by the user
        training = call_spec.user_arguments_dict.get("training", None)
        if training is None:
            # Wasn't passed explicitly: use context value
            training = call_context.training
            if training is None:
                # Get signature default value; else False
                training = call_spec.arguments_dict.get("training", False)
        call_context.training = training
        if self._call_has_training_arg():
            kwargs["training"] = training

        ##############################
        # 6. Populate mask argument(s)
        if self.supports_masking:
            if len(call_spec.tensor_arguments_dict) == 1:
                if (
                    "mask" in call_spec.argument_names
                    and call_spec.arguments_dict["mask"] is None
                ):
                    arg_name = list(call_spec.tensor_arguments_dict.keys())[0]
                    only_tensor_arg = call_spec.tensor_arguments_dict[arg_name]
                    mask = nest.map_structure(
                        lambda x: getattr(x, "_keras_mask", None),
                        only_tensor_arg,
                    )
                    kwargs["mask"] = mask
            elif len(call_spec.tensor_arguments_dict) > 1:
                for k, v in call_spec.tensor_arguments_dict.items():
                    expected_mask_arg_name = f"{k}_mask"
                    if expected_mask_arg_name in call_spec.argument_names:
                        if (
                            call_spec.arguments_dict[expected_mask_arg_name]
                            is None
                        ):
                            mask = nest.map_structure(
                                lambda x: getattr(x, "_keras_mask", None), v
                            )
                            kwargs[expected_mask_arg_name] = mask

        ####################
        # 7. Call the layer.
        try:
            with backend.name_scope(self.name):
                if self.autocast and self.compute_dtype != self.variable_dtype:
                    # For mixed precision, we automatically cast layer variables
                    # (float ones only) to the compute dtype upon access.
                    with backend.AutocastScope(self.compute_dtype):
                        outputs = super().__call__(*args, **kwargs)
                else:
                    outputs = super().__call__(*args, **kwargs)
                if not self.built:
                    self.built = True
                # Record activity regularizer loss.
                if self.activity_regularizer is not None:
                    for output in nest.flatten(outputs):
                        if backend.is_tensor(output):
                            self.add_loss(self.activity_regularizer(output))

            if self.supports_masking:
                # Set masks on outputs,
                # provided only the first positional input arg and its mask.
                # TODO: consider extending this to all args and kwargs.
                previous_mask = getattr(
                    call_spec.first_arg, "_keras_mask", None
                )
                self._set_mask_metadata(
                    call_spec.first_arg, outputs, previous_mask
                )
        finally:
            # Destroy call context if we created it
            self._maybe_reset_call_context()
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @traceback_utils.filter_traceback
    def stateless_call(
        self,
        trainable_variables,
        non_trainable_variables,
        *args,
        return_losses=False,
        **kwargs,
    ):
        """Call the layer without any side effects.

        Args:
            trainable_variables: List of trainable variables of the model.
            non_trainable_variables: List of non-trainable variables of the
                model.
            *args: Positional argumets to be passed to `call()`.
            return_losses: If `True`, `stateless_call()` will return the list of
                losses created during `call()` as part of its return values.
            **kwargs: Keyword arguments to be passed to `call()`.

        Returns:
            A tuple. By default, returns `(outputs, non_trainable_variables)`.
                If `return_losses = True`, then returns
                `(outputs, non_trainable_variables, losses)`.

        Note: `non_trainable_variables` include not only non-trainable weights
        such as `BatchNormalization` statistics, but also RNG seed state
        (if there are any random operations part of the layer, such as dropout),
        and `Metric` state (if there are any metrics attached to the layer).
        These are all elements of state of the layer.

        Example:

        ```python
        model = ...
        data = ...
        trainable_variables = model.trainable_variables
        non_trainable_variables = model.non_trainable_variables
        # Call the model with zero side effects
        outputs, non_trainable_variables = model.stateless_call(
            trainable_variables,
            non_trainable_variables,
            data,
        )
        # Attach the updated state to the model
        # (until you do this, the model is still in its pre-call state).
        for ref_var, value in zip(
            model.non_trainable_variables, non_trainable_variables
        ):
            ref_var.assign(value)
        ```
        """
        self._check_super_called()

        if not self.built:
            raise ValueError(
                f"To call stateless_call, {self.__class__.__name__} must be "
                "built (i.e. its variables must have been already created). "
                "You can build it by calling it on some data."
            )
        if len(trainable_variables) != len(self.trainable_variables):
            raise ValueError(
                "Argument `trainable_variables` must be a list of tensors "
                "corresponding 1:1 to "
                f"{self.__class__.__name__}().trainable_variables. "
                f"Received list with length {len(trainable_variables)}, "
                f"but expected {len(self.trainable_variables)} variables."
            )
        if len(non_trainable_variables) != len(self.non_trainable_variables):
            raise ValueError(
                "Argument `non_trainable_variables` must be a list of tensors "
                "corresponding 1:1 to "
                f"{self.__class__.__name__}().non_trainable_variables. "
                f"Received list with length {len(non_trainable_variables)}, "
                f"but expected {len(self.non_trainable_variables)} variables."
            )

        # Gather variable mapping
        trainable_mapping = zip(self.trainable_variables, trainable_variables)
        non_trainable_mapping = zip(
            self.non_trainable_variables, non_trainable_variables
        )
        mapping = list(trainable_mapping) + list(non_trainable_mapping)

        # Call in stateless scope
        with backend.StatelessScope(
            state_mapping=mapping, collect_losses=return_losses
        ) as scope:
            outputs = self.call(*args, **kwargs)

        # Gather updated non-trainable variables
        non_trainable_variables = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                non_trainable_variables.append(new_v)
            else:
                non_trainable_variables.append(v)

        if return_losses:
            return outputs, non_trainable_variables, scope.losses[:]
        return outputs, non_trainable_variables

    def compute_output_spec(self, *args, **kwargs):
        if utils.is_default(self.compute_output_shape):
            return super().compute_output_spec(*args, **kwargs)
        else:
            # Use compute_output_shape() to return the right output spec
            call_spec = CallSpec(self.call, args, kwargs)
            shapes_dict = get_shapes_dict(
                self.compute_output_shape, call_spec, self.__class__
            )
            if len(shapes_dict) == 1:
                # Single arg: pass it positionally
                input_shape = tuple(shapes_dict.values())[0]
                output_shape = self.compute_output_shape(input_shape)
            else:
                # More than one shape: pass them by name.
                output_shape = self.compute_output_shape(**shapes_dict)

            if (
                isinstance(output_shape, list)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                output_shape = tuple(output_shape)
            if not isinstance(output_shape, (list, tuple, dict)):
                try:
                    output_shape = tuple(output_shape)
                except:
                    raise ValueError(
                        "Method `compute_output_shape()` of layer "
                        f"{self.__class__.__name__} is returning "
                        "a type that cannot be interpreted as a shape. "
                        "It should return a shape tuple. "
                        f"Received: {output_shape}"
                    )
            if (
                isinstance(output_shape, tuple)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                return KerasTensor(output_shape, dtype=self.compute_dtype)
            # Case: nested. Could be a tuple/list of shapes, or a dict of
            # shapes. Could be deeply nested.
            return map_shape_structure(
                lambda s: KerasTensor(s, dtype=self.compute_dtype), output_shape
            )

    @utils.default
    def compute_output_shape(self, *args, **kwargs):
        return NotImplementedError

    def add_loss(self, loss):
        # Eager only.
        losses = nest.flatten(loss)
        for x in losses:
            if not backend.is_tensor(x):
                raise ValueError(
                    "`add_loss()` can only be called from inside `build()` or "
                    f"`call()`, on a tensor input. Received invalid value: {x}"
                )
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in losses:
                    scope.add_loss(loss)
        else:
            self._losses.extend(losses)

    @property
    def losses(self):
        losses = self._losses[:]
        for layer in self._layers:
            losses.extend(layer._losses)
        weight_regularization_losses = []
        for v in self.trainable_weights:
            regularizer = getattr(v, "regularizer", None)
            if regularizer:
                weight_regularization_losses.append(regularizer(v))
        losses.extend(weight_regularization_losses)
        return losses

    def save_own_variables(self, store):
        """Saves the state of the layer.

        You can override this method to take full control of how the state of
        the layer is saved upon calling `model.save()`.

        Args:
            store: Dict where the state of the model will be saved.
        """
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            store[f"{i}"] = np.array(v)

    def load_own_variables(self, store):
        """Loads the state of the layer.

        You can override this method to take full control of how the state of
        the layer is loaded upon calling `keras.models.load_model()`.

        Args:
            store: Dict from which the state of the model will be loaded.
        """
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer. In most cases, "
                    "this indicates that you need to implement the "
                    "`def build_from_config(self, config)` method "
                    "on the layer. "
                    "You might also want to implement the method "
                    "that generates the config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
        for i, v in enumerate(all_vars):
            v.assign(store[f"{i}"])

    def _clear_losses(self):
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in scope.losses:
                    if x in self._losses:
                        scope.losses.remove(x)
        self._losses.clear()
        for layer in self._layers:
            layer._clear_losses()

    def _track_variable(self, variable):
        if variable.trainable:
            self._trainable_variables.append(variable)
            # Prevent double-tracking
            self._tracker.stored_ids["trainable_variables"].add(id(variable))
        else:
            self._non_trainable_variables.append(variable)
            # Prevent double-tracking
            self._tracker.stored_ids["non_trainable_variables"].add(
                id(variable)
            )

    def add_metric(self):
        # Permanently disabled
        raise NotImplementedError

    def count_params(self):
        """Count the total number of scalars composing the weights.

        Returns:
            An integer count.
        """
        if not self.built:
            raise ValueError(
                "You tried to call `count_params` "
                f"on layer '{self.name}', "
                "but the layer isn't built. "
                "You can build it manually via: "
                f"`layer.build(input_shape)`."
            )
        return summary_utils.count_params(self.weights)

    def _maybe_build(self, call_spec):
        if not self.built:
            shapes_dict = get_shapes_dict(self.build, call_spec, self.__class__)
            self._build_shapes_dict = shapes_dict
            failure = False
            if len(shapes_dict) == 1:
                # Single arg: pass it positionally
                input_shape = tuple(shapes_dict.values())[0]
                with backend.name_scope(self.name):
                    if utils.is_default(
                        self.build
                    ) and might_have_unbuilt_state(self):
                        status = self._build_by_run_for_single_pos_arg(
                            input_shape
                        )
                        if not status:
                            failure = True
                    else:
                        self.build(input_shape)
            else:
                with backend.name_scope(self.name):
                    if utils.is_default(self.build):
                        if might_have_unbuilt_state(self):
                            status = self._build_by_run_for_kwargs(shapes_dict)
                            if not status:
                                failure = True
                    else:
                        self.build(**shapes_dict)
            if failure:
                if call_spec.eager:
                    # Will let the actual eager call do the state-building
                    return
                raise ValueError(
                    f"Layer '{self.name}' looks like it has "
                    "unbuilt state, but Keras is not able to "
                    "trace the layer `call()` in order to "
                    "build it automatically. You must implement "
                    "the `def build(self, input_shape)` method on your "
                    "layer. It should create all variables used by the "
                    "layer (e.g. by calling `layer.build()` on all its "
                    "children layers)."
                )
            self.built = True

            # Check input spec again (after build, since self.input_spec
            # may have been updated
            self._assert_input_compatibility(call_spec.first_arg)

    def _build_by_run_for_single_pos_arg(self, input_shape):
        # Case: all inputs are in the first arg (possibly nested).
        if is_shape_tuple(input_shape):
            input_shape = tuple(input_shape)
        if isinstance(input_shape, list):
            input_tensors = [
                backend.KerasTensor(shape) for shape in input_shape
            ]
        elif isinstance(input_shape, dict):
            input_tensors = {
                k: backend.KerasTensor(shape)
                for k, shape in input_shape.items()
            }
        else:
            input_tensors = backend.KerasTensor(input_shape)
        try:
            backend.compute_output_spec(self.call, input_tensors)
            return True
        except:
            return False

    def _build_by_run_for_kwargs(self, shapes_dict):
        # Case: inputs were recorded as multiple keyword arguments.
        if all(is_shape_tuple(s) for s in shapes_dict.values()):
            # Case: all input keyword arguments were plain tensors.
            input_tensors = {
                # We strip the `_shape` suffix to recover kwarg names.
                k.removesuffix("_shape"): backend.KerasTensor(shape)
                for k, shape in shapes_dict.items()
            }
            try:
                backend.compute_output_spec(self.call, **input_tensors)
                return True
            except:
                return False
        else:
            # Not supported: nested input keyword arguments.
            return False

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name}, built={self.built}>"
        )

    def __str__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name}, built={self.built}>"
        )

    def __setattr__(self, name, value):
        # Prevent users from attaching state to the
        # layer before `super()` is called -- since that
        # state would silently not be tracked.
        if name != "_lock":
            self._check_super_called()
        # Track Variables, Layers, Metrics, SeedGenerators.
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _check_super_called(self):
        if getattr(self, "_lock", True):
            raise RuntimeError(
                f"In layer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. Go add it!"
            )

    def _assert_input_compatibility(self, arg_0):
        if self.input_spec:
            input_spec.assert_input_compatibility(
                self.input_spec, arg_0, layer_name=self.name
            )

    def _call_has_training_arg(self):
        return "training" in self._call_signature_parameters

    def _call_has_mask_arg(self):
        return "mask" in self._call_signature_parameters

    def _get_call_context(self):
        """Returns currently active `CallContext`."""
        layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if layer_call_ctx is None:
            # Enter new call context.
            layer_call_ctx = CallContext(entry_layer=self)
            global_state.set_global_attribute(
                "current_call_ctx", layer_call_ctx
            )
            self._clear_losses()
        return layer_call_ctx

    def _maybe_reset_call_context(self):
        layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if layer_call_ctx is None or layer_call_ctx.entry_layer == self:
            global_state.set_global_attribute("current_call_ctx", None)

    def _flatten_layers(self, include_self=True, recursive=True):
        layers = []
        if include_self:
            layers.append(self)
        seen_object_ids = set()
        deque = collections.deque(self._layers)
        while deque:
            layer = deque.popleft()
            if id(layer) in seen_object_ids:
                continue
            seen_object_ids.add(id(layer))
            layers.append(layer)
            # Introspect recursively through sublayers.
            if recursive:
                deque.extendleft(layer._layers)
        return layers

    def _set_mask_metadata(self, inputs, outputs, previous_mask):
        flat_outputs = nest.flatten(outputs)

        mask_already_computed = all(
            getattr(x, "_keras_mask", None) is not None for x in flat_outputs
        )
        if mask_already_computed:
            return

        output_masks = self.compute_mask(inputs, previous_mask)
        if output_masks is None:
            return

        flat_masks = nest.flatten(output_masks)
        for tensor, mask in zip(flat_outputs, flat_masks):
            if getattr(tensor, "_keras_mask", None) is None:
                try:
                    tensor._keras_mask = mask
                except AttributeError:
                    # It's a C type.
                    pass

    @python_utils.default
    def get_config(self):
        base_config = super().get_config()
        config = {
            "trainable": self.trainable,
            "dtype": self.dtype_policy.name,
        }
        return {**base_config, **config}


def is_backend_tensor_or_symbolic(x):
    return backend.is_tensor(x) or isinstance(x, backend.KerasTensor)


class CallSpec:
    def __init__(self, call_fn, args, kwargs):
        sig = inspect.signature(call_fn)
        bound_args = sig.bind(*args, **kwargs)
        self.user_arguments_dict = {
            k: v for k, v in bound_args.arguments.items()
        }
        bound_args.apply_defaults()
        arg_dict = {}
        arg_names = []
        tensor_arg_dict = {}
        tensor_args = []
        tensor_arg_names = []
        nested_tensor_arg_names = []
        for name, value in bound_args.arguments.items():
            arg_dict[name] = value
            arg_names.append(name)
            if is_backend_tensor_or_symbolic(value):
                tensor_args.append(value)
                tensor_arg_names.append(name)
                tensor_arg_dict[name] = value
            elif nest.is_nested(value):
                flat_values = nest.flatten(value)
                if all(is_backend_tensor_or_symbolic(x) for x in flat_values):
                    tensor_args.append(value)
                    tensor_arg_names.append(name)
                    tensor_arg_dict[name] = value
                    nested_tensor_arg_names.append(name)
                elif any(is_backend_tensor_or_symbolic(x) for x in flat_values):
                    raise ValueError(
                        "In a nested call() argument, "
                        "you cannot mix tensors and non-tensors. "
                        "Received invalid mixed argument: "
                        f"{name}={value}"
                    )
        self.arguments_dict = arg_dict
        self.argument_names = arg_names
        self.tensor_arguments_dict = tensor_arg_dict
        self.tensor_arguments_names = tensor_arg_names
        self.nested_tensor_argument_names = nested_tensor_arg_names
        self.first_arg = arg_dict[arg_names[0]]
        if all(
            backend.is_tensor(x) for x in self.tensor_arguments_dict.values()
        ):
            self.eager = True
        else:
            self.eager = False


def get_arguments_dict(fn, args, kwargs):
    """Return a dict mapping argument names to their values."""
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    arg_dict = {}
    for name, value in bound_args.arguments.items():
        arg_dict[name] = value
    return arg_dict


def get_shapes_dict(target_fn, call_spec, cls):
    """Convert the call() arguments dict into a dict of input shape arguments.

    Example:

    ```
    >>> get_shapes_dict(self.build, call_spec, cls)
    {"input_a_shape": (2, 3)}
    ```
    """
    expected_names = check_shapes_signature(target_fn, call_spec, cls)
    shapes_dict = {}
    for k, v in call_spec.tensor_arguments_dict.items():
        if k == "mask" or k.startswith("mask_"):
            # Do not include mask tensors in shapes dict
            continue
        if k == "kwargs" or k == "args":
            # Do not include catch-alls in shapes dict
            continue
        if expected_names is not None and f"{k}_shape" not in expected_names:
            continue
        if k in call_spec.nested_tensor_argument_names:
            shapes_dict[f"{k}_shape"] = nest.map_structure(
                lambda x: backend.standardize_shape(x.shape), v
            )
        else:
            shapes_dict[f"{k}_shape"] = backend.standardize_shape(v.shape)
    return shapes_dict


def check_shapes_signature(target_fn, call_spec, cls):
    """Asserts that the argument names in `target_fn` match arguments in `call`.

    We use this to check that `build()` and `compute_output_shape()` arguments
    align with `call()` arguments.

    For instance if `build()` has the signature
    `def build(self, a_shape, b_shape)` we expect `call()` to accept the
    arguments `a` and `b`.

    When there is a single argument accepted by `target_fn`, we do allow any
    name and do not check the call signature.

    Returns:
        The list of arguments names expected by the `target_fn` or
        `None` if any passed name is acceptable.
    """
    if utils.is_default(target_fn):
        return None
    sig = inspect.signature(target_fn)
    expected_names = []
    for name, param in sig.parameters.items():
        if param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.POSITIONAL_ONLY,
            param.KEYWORD_ONLY,
        ):
            expected_names.append(name)
    if len(expected_names) == 1:
        return None
    for name in expected_names:
        method_name = target_fn.__name__
        error_preamble = (
            f"For a `{method_name}()` method with more than one argument, all "
            "arguments should have a `_shape` suffix and match an argument "
            f"from `call()`. E.g. `{method_name}(self, foo_shape, bar_shape)` "
            "would match `call(self, foo, bar)`."
        )
        if not name.endswith("_shape"):
            raise ValueError(
                f"{error_preamble} For layer '{cls.__name__}', "
                f"Received `{method_name}()` argument "
                f"`{name}`, which does not end in `_shape`."
            )
        expected_call_arg = name.removesuffix("_shape")
        if expected_call_arg not in call_spec.arguments_dict:
            raise ValueError(
                f"{error_preamble} For layer '{cls.__name__}', "
                f"received `{method_name}()` argument "
                f"`{name}`, but `call()` does not have argument "
                f"`{expected_call_arg}`."
            )
    return expected_names


class CallContext:
    def __init__(self, entry_layer):
        self.entry_layer = entry_layer
        self.training = None


def is_shape_tuple(s):
    return isinstance(s, (list, tuple)) and all(
        d is None or isinstance(d, int) for d in s
    )


def might_have_unbuilt_state(layer):
    return any(not lr.built for lr in layer._layers)
