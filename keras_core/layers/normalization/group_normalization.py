from keras_core import constraints
from keras_core import initializers
from keras_core import operations as ops
from keras_core import regularizers
from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.GroupNormalization")
class GroupNormalization(Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization.

    Args:
        groups: Integer, the number of groups for Group Normalization. Can be in
            the range [1, N] where N is the input dimension. The input dimension
            must be divisible by the number of groups. Defaults to `32`.
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. `-1` is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero. Defaults
            to 1e-3
        center: If True, add offset of `beta` to normalized tensor. If False,
            `beta` is ignored. Defaults to `True`.
        scale: If True, multiply by `gamma`. If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight. None by
            default.
        gamma_regularizer: Optional regularizer for the gamma weight. None by
            default.
        beta_constraint: Optional constraint for the beta weight. None by default.
        gamma_constraint: Optional constraint for the gamma weight. None by
            default.  Input shape: Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis) when using this
            layer as the first layer in a model.  Output shape: Same shape as input.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).

    Reference:
    
    - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        # tf_utils.validate_axis(self.axis, input_shape)
        # if isinstance(self.axis, int):
        #     self.axis = [self.axis]
        # else:
        #     self.axis = list(self.axis)
        # for idx, x in enumerate(self.axis):
        #     if x < 0:
        #         self.axis[idx] = len(input_shape) + x

        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}."
            )

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim})."
            )

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim})."
            )

        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        input_shape = inputs.shape

        reshaped_inputs = self._reshape_into_groups(inputs)

        normalized_inputs = self._apply_normalization(
            reshaped_inputs, input_shape
        )

        return ops.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs):
        input_shape = inputs.shape
        group_shape = [input_shape[i] for i in range(len(input_shape))]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = ops.stack(group_shape)
        reshaped_inputs = ops.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, len(reshaped_inputs.shape)))

        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean = ops.mean(reshaped_inputs, axis=group_reduction_axes, keepdims=True)
        variance = ops.var(reshaped_inputs, axis=group_reduction_axes, keepdims=True)

        gamma, beta = self._get_reshaped_weights(input_shape)

        # Compute the batch normalization.
        inv = 1 / ops.sqrt(variance + self.epsilon)
        if gamma is not None:
            inv *= gamma

        x = beta - mean * inv if beta is not None else -mean * inv
        normalized_inputs = reshaped_inputs * ops.cast(inv, reshaped_inputs.dtype) + ops.cast(
            x, reshaped_inputs.dtype
        )

        normalized_inputs = ops.cast(normalized_inputs, reshaped_inputs.dtype)

        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = ops.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = ops.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):

        broadcast_shape = [1] * len(input_shape)

        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)

        return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
