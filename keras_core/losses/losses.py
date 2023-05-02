import warnings

from keras_core import backend
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.losses.loss import Loss
from keras_core.losses.loss import squeeze_to_same_rank
from keras_core.saving import serialization_lib
from keras_core.utils.numerical_utils import normalize


class LossFunctionWrapper(Loss):
    def __init__(
        self, fn, reduction="sum_over_batch_size", name=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {"fn": serialization_lib.serialize_keras_object(self.fn)}
        config.update(serialization_lib.serialize_keras_object(self._fn_kwargs))
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_keras_object(config)
        return cls(**config)


@keras_core_export("keras_core.losses.MeanSquaredError")
class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error"
    ):
        super().__init__(mean_squared_error, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.MeanAbsoluteError")
class MeanAbsoluteError(LossFunctionWrapper):
    """Computes the mean of absolute difference between labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error"
    ):
        super().__init__(mean_absolute_error, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_percentage_error",
    ):
        super().__init__(
            mean_absolute_percentage_error, reduction=reduction, name=name
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_logarithmic_error",
    ):
        super().__init__(
            mean_squared_logarithmic_error, reduction=reduction, name=name
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.CosineSimilarity")
class CosineSimilarity(LossFunctionWrapper):
    """Computes the cosine similarity between `y_true` & `y_pred`.

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
    will be 0 regardless of the proximity between predictions and targets.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        axis: The axis along which the cosine similarity is computed
            (the features axis). Defaults to -1.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self,
        axis=-1,
        reduction="sum_over_batch_size",
        name="cosine_similarity",
    ):
        super().__init__(
            cosine_similarity, reduction=reduction, name=name, axis=axis
        )


@keras_core_export("keras_core.losses.Huber")
class Huber(LossFunctionWrapper):
    """Computes the Huber loss between `y_true` & `y_pred`.

    For each value x in `error = y_true - y_pred`:

    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Args:
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        reduction: Type of reduction to apply to loss. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`. Defaults to
            `"sum_over_batch_size"`.
        name: Optional name for the instance.
    """

    def __init__(
        self,
        delta=1.0,
        reduction="sum_over_batch_size",
        name="huber_loss",
    ):
        super().__init__(huber, name=name, reduction=reduction, delta=delta)


@keras_core_export("keras_core.losses.LogCosh")
class LogCosh(LossFunctionWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    `logcosh = log((exp(x) + exp(-x))/2)`,
    where x is the error `y_pred - y_true`.

    Args:
        reduction: Type of reduction to apply to loss. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`. Defaults to
            `"sum_over_batch_size"`.
        name: Optional name for the instance.
    """

    def __init__(self, reduction="sum_over_batch_size", name="log_cosh"):
        super().__init__(log_cosh, name=name, reduction=reduction)


@keras_core_export("keras_core.losses.Hinge")
class Hinge(LossFunctionWrapper):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(1 - y_true * y_pred, 0)
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(self, reduction="sum_over_batch_size", name="hinge"):
        super().__init__(hinge, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.SquaredHinge")
class SquaredHinge(LossFunctionWrapper):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = square(maximum(1 - y_true * y_pred, 0))
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge"):
        super().__init__(squared_hinge, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.CategoricalHinge")
class CategoricalHinge(LossFunctionWrapper):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge"
    ):
        super().__init__(categorical_hinge, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.KLDivergence")
class KLDivergence(LossFunctionWrapper):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_true * log(y_true / y_pred)
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(self, reduction="sum_over_batch_size", name="kl_divergence"):
        super().__init__(kl_divergence, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.Poisson")
class Poisson(LossFunctionWrapper):
    """Computes the Poisson loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_pred - y_true * log(y_pred)
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(self, reduction="sum_over_batch_size", name="poisson"):
        super().__init__(poisson, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.BinaryCrossentropy")
class BinaryCrossentropy(LossFunctionWrapper):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
      floating-point value which either represents a
      [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
      when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
      `from_logits=False`).

    Args:
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).
        label_smoothing: Float in range [0, 1]. When 0, no smoothing occurs.
            When > 0, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards 0.5. Larger values of
            `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.

    Examples:

    **Recommended Usage:** (set `from_logits=True`)

    With `compile()` API:

    ```python
    model.compile(
        loss=keras_core.losses.BinaryCrossentropy(from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = [0, 1, 0, 0]
    >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
    >>> bce = keras_core.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.865

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> bce = keras_core.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.865
    >>> # Using 'sample_weight' attribute
    >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.243
    >>> # Using 'sum' reduction` type.
    >>> bce = keras_core.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction="sum")
    >>> bce(y_true, y_pred)
    1.730
    >>> # Using 'none' reduction type.
    >>> bce = keras_core.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction=None)
    >>> bce(y_true, y_pred)
    array([0.235, 1.496], dtype=float32)

    **Default Usage:** (set `from_logits=False`)

    >>> # Make the following updates to the above "Recommended Usage" section
    >>> # 1. Set `from_logits=False`
    >>> keras_core.losses.BinaryCrossentropy() # OR ...('from_logits=False')
    >>> # 2. Update `y_pred` to use probabilities instead of logits
    >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_crossentropy",
    ):
        super().__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
        }


@keras_core_export("keras_core.losses.BinaryFocalCrossentropy")
class BinaryFocalCrossentropy(LossFunctionWrapper):
    """Computes focal cross-entropy loss between true labels and predictions.

    Binary cross-entropy loss is often used for binary (0 or 1) classification
    tasks. The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
      floating-point value which either represents a
      [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
      when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
      `from_logits=False`).

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a "focal factor" to down-weight easy examples and focus more
    on hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma=0`, this function is
    equivalent to the binary crossentropy loss.

    Args:
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in reference [Lin et al., 2018](
            https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
            `1.0 - alpha`.
        gamma: A focusing parameter used to compute the focal factor, default is
            `2.0` as mentioned in the reference
            [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
        label_smoothing: Float in `[0, 1]`. When `0`, no smoothing occurs.
            When > `0`, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards `0.5`.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.

    Examples:

    With the `compile()` API:

    ```python
    model.compile(
        loss=keras_core.losses.BinaryFocalCrossentropy(
            gamma=2.0, from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = [0, 1, 0, 0]
    >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...    gamma=2, from_logits=True)
    >>> loss(y_true, y_pred)
    0.691

    >>> # Apply class weight
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=2, from_logits=True)
    >>> loss(y_true, y_pred)
    0.51

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     gamma=3, from_logits=True)
    >>> loss(y_true, y_pred)
    0.647

    >>> # Apply class weight
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...      apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred)
    0.482

    >>> # Using 'sample_weight' attribute with focal effect
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.133

    >>> # Apply class weight
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...      apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.097

    >>> # Using 'sum' reduction` type.
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     gamma=4, from_logits=True,
    ...     reduction="sum")
    >>> loss(y_true, y_pred)
    1.222

    >>> # Apply class weight
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=4, from_logits=True,
    ...     reduction="sum")
    >>> loss(y_true, y_pred)
    0.914

    >>> # Using 'none' reduction type.
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     gamma=5, from_logits=True,
    ...     reduction=None)
    >>> loss(y_true, y_pred)
    array([0.0017 1.1561], dtype=float32)

    >>> # Apply class weight
    >>> loss = keras_core.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=5, from_logits=True,
    ...     reduction=None)
    >>> loss(y_true, y_pred)
    array([0.0004 0.8670], dtype=float32)
    """

    def __init__(
        self,
        apply_class_balancing=False,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_focal_crossentropy",
    ):
        super().__init__(
            binary_focal_crossentropy,
            apply_class_balancing=apply_class_balancing,
            alpha=alpha,
            gamma=gamma,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
            "apply_class_balancing": self.apply_class_balancing,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }


@keras_core_export("keras_core.losses.CategoricalCrossentropy")
class CategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided in a `one_hot` representation. If
    you want to provide labels as integers, please use
    `SparseCategoricalCrossentropy` loss.  There should be `# classes` floating
    point values per feature.

    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to -1.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.

    Examples:

    Standalone usage:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = keras_core.losses.CategoricalCrossentropy()
    >>> cce(y_true, y_pred)
    1.177

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.814

    >>> # Using 'sum' reduction type.
    >>> cce = keras_core.losses.CategoricalCrossentropy(
    ...     reduction="sum")
    >>> cce(y_true, y_pred)
    2.354

    >>> # Using 'none' reduction type.
    >>> cce = keras_core.losses.CategoricalCrossentropy(
    ...     reduction=None)
    >>> cce(y_true, y_pred)
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss=keras_core.losses.CategoricalCrossentropy())
    ```
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_crossentropy",
    ):
        super().__init__(
            categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
        }


@keras_core_export("keras_core.losses.CategoricalFocalCrossentropy")
class CategoricalFocalCrossentropy(LossFunctionWrapper):
    """Computes the alpha balanced focal crossentropy loss.

    Use this crossentropy loss function when there are two or more label
    classes and if you want to handle class imbalance without using
    `class_weights`. We expect labels to be provided in a `one_hot`
    representation.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL)
    is as follows:

    `FL(p_t) = (1 - p_t) ** gamma * log(p_t)`

    where `p_t` is defined as follows:
    `p_t = output if y_true == 1, else 1 - output`

    `(1 - p_t) ** gamma` is the `modulating_factor`, where `gamma` is a focusing
    parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
    `gamma` reduces the importance given to simple examples in a smooth manner.

    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    `FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)`

    where `alpha` is the weight factor for the classes. If `alpha` = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.

    The formula above can be generalized to:
    `FL(p_t) = alpha * (1 - p_t) ** gamma * CrossEntropy(y_true, y_pred)`

    where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

    Extending this to multi-class case is straightforward:
    `FL(p_t) = alpha * (1 - p_t) ** gamma * CategoricalCE(y_true, y_pred)`

    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `(batch_size, num_classes)`.

    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        from_logits: Whether `output` is expected to be a logits tensor. By
            default, we consider that `output` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to -1.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.

    Examples:

    Standalone usage:

    >>> y_true = [[0., 1., 0.], [0., 0., 1.]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = keras_core.losses.CategoricalFocalCrossentropy()
    >>> cce(y_true, y_pred)
    0.23315276

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.1632

    >>> # Using 'sum' reduction type.
    >>> cce = keras_core.losses.CategoricalFocalCrossentropy(
    ...     reduction="sum")
    >>> cce(y_true, y_pred)
    0.46631

    >>> # Using 'none' reduction type.
    >>> cce = keras_core.losses.CategoricalFocalCrossentropy(
    ...     reduction=None)
    >>> cce(y_true, y_pred)
    array([3.2058331e-05, 4.6627346e-01], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='adam',
                  loss=keras_core.losses.CategoricalFocalCrossentropy())
    ```
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_focal_crossentropy",
    ):
        """Initializes `CategoricalFocalCrossentropy` instance."""
        super().__init__(
            categorical_focal_crossentropy,
            alpha=alpha,
            gamma=gamma,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis
        self.alpha = alpha
        self.gamma = gamma

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }


@keras_core_export("keras_core.losses.SparseCategoricalCrossentropy")
class SparseCategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label
    classes.  We expect labels to be provided as integers. If you want to
    provide labels using `one-hot` representation, please use
    `CategoricalCrossentropy` loss.  There should be `# classes` floating point
    values per feature for `y_pred` and a single floating point value per
    feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.

    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Suuported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.

    Examples:

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> scce = keras_core.losses.SparseCategoricalCrossentropy()
    >>> scce(y_true, y_pred)
    1.177

    >>> # Calling with 'sample_weight'.
    >>> scce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.814

    >>> # Using 'sum' reduction type.
    >>> scce = keras_core.losses.SparseCategoricalCrossentropy(
    ...     reduction="sum")
    >>> scce(y_true, y_pred)
    2.354

    >>> # Using 'none' reduction type.
    >>> scce = keras_core.losses.SparseCategoricalCrossentropy(
    ...     reduction=None)
    >>> scce(y_true, y_pred)
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss=keras_core.losses.SparseCategoricalCrossentropy())
    ```
    """

    def __init__(
        self,
        from_logits=False,
        reduction="sum_over_batch_size",
        name="sparse_categorical_crossentropy",
    ):
        super().__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
        )
        self.from_logits = from_logits

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
        }


def convert_binary_labels_to_hinge(y_true):
    """Converts binary labels into -1/1 for hinge loss/metric calculation."""
    are_zeros = ops.equal(y_true, 0)
    are_ones = ops.equal(y_true, 1)
    is_binary = ops.all((ops.logical_or(are_zeros, are_ones)))

    def _convert_binary_labels():
        # Convert the binary labels to -1 or 1.
        return 2.0 * y_true - 1.0

    def _return_labels_unconverted():
        # Returns the labels unchanged if they are non-binary
        return y_true

    updated_y_true = ops.cond(
        is_binary, _convert_binary_labels, _return_labels_unconverted
    )
    return updated_y_true


@keras_core_export(
    [
        "keras_core.metrics.hinge",
        "keras_core.losses.hinge",
    ]
)
def hinge(y_true, y_pred):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)
    ```

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided they will be converted
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.hinge(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, dtype=y_pred.dtype)
    y_true = ops.convert_to_tensor(y_true)
    y_true = convert_binary_labels_to_hinge(y_true)
    return ops.mean(ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.squared_hinge",
        "keras_core.losses.squared_hinge",
    ]
)
def squared_hinge(y_true, y_pred):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
    ```

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided we will convert them
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.squared_hinge(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    y_true = convert_binary_labels_to_hinge(y_true)
    return ops.mean(
        ops.square(ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1
    )


@keras_core_export(
    [
        "keras_core.metrics.categorical_hinge",
        "keras_core.losses.categorical_hinge",
    ]
)
def categorical_hinge(y_true, y_pred):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        y_true: The ground truth values. `y_true` values are expected to be
            either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
            shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 3, size=(2,))
    >>> y_true = np.eye(np.max(y_true) + 1)[y_true]
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.categorical_hinge(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    pos = ops.sum(y_true * y_pred, axis=-1)
    neg = ops.max((1.0 - y_true) * y_pred, axis=-1)
    zero = ops.cast(0.0, y_pred.dtype)
    return ops.maximum(neg - pos + 1.0, zero)


@keras_core_export(
    [
        "keras_core.metrics.mean_squared_error",
        "keras_core.losses.mean_squared_error",
    ]
)
def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred), axis=-1)
    ```

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_squared_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    return ops.mean(ops.square(y_true - y_pred), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.mean_absolute_error",
        "keras_core.losses.mean_absolute_error",
    ]
)
def mean_absolute_error(y_true, y_pred):
    """Computes the mean absolute error between labels and predictions.

    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_absolute_error(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    return ops.mean(ops.abs(y_true - y_pred), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.mean_absolute_percentage_error",
        "keras_core.losses.mean_absolute_percentage_error",
    ]
)
def mean_absolute_percentage_error(y_true, y_pred):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
    ```

    Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
    where `epsilon = keras_core.backend.epsilon()`
    (default to `1e-7`).

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute percentage error values with shape = `[batch_size, d0, ..
        dN-1]`.

    Example:

    >>> y_true = np.random.random(size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_absolute_percentage_error(y_true, y_pred)
    """
    epsilon = ops.convert_to_tensor(backend.epsilon())
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    diff = ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), epsilon))
    return 100.0 * ops.mean(diff, axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.mean_squared_logarithmic_error",
        "keras_core.losses.mean_squared_logarithmic_error",
    ]
)
def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
    ```

    Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
    values and 0 values will be replaced with `keras_core.backend.epsilon()`
    (default to `1e-7`).

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared logarithmic error values with shape = `[batch_size, d0, ..
        dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_squared_logarithmic_error(y_true, y_pred)
    """
    epsilon = ops.convert_to_tensor(backend.epsilon())
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    first_log = ops.log(ops.maximum(y_pred, epsilon) + 1.0)
    second_log = ops.log(ops.maximum(y_true, epsilon) + 1.0)
    return ops.mean(ops.square(first_log - second_log), axis=-1)


@keras_core_export("keras_core.losses.cosine_similarity")
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Formula:
    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine
    similarity will be 0 regardless of the proximity between predictions
    and targets.

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to -1.

    Returns:
        Cosine similarity tensor.

    Example:

    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> loss = keras_core.losses.cosine_similarity(y_true, y_pred, axis=-1)
    [-0., -0.99999994, 0.99999994]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return -ops.sum(y_true * y_pred, axis=axis)


@keras_core_export(["keras_core.losses.huber", "keras_core.metrics.huber"])
def huber(y_true, y_pred, delta=1.0):
    """Computes Huber loss value.

    Formula:
    ```python
    for x in error:
        if abs(x) <= delta:
            loss.append(0.5 * x^2)
        elif abs(x) > delta:
            loss.append(delta * abs(x) - 0.5 * delta^2)

    loss = mean(loss, axis=-1)
    ```
    See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras_core.losses.huber(y_true, y_pred)
    0.155


    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear. Defaults to 1.

    Returns:
        Tensor with one scalar loss entry per sample.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    delta = ops.convert_to_tensor(delta)
    error = ops.subtract(y_pred, y_true)
    abs_error = ops.abs(error)
    half = ops.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return ops.mean(
        ops.where(
            abs_error <= delta,
            half * ops.square(error),
            delta * abs_error - half * ops.square(delta),
        ),
        axis=-1,
    )


@keras_core_export(
    ["keras_core.losses.log_cosh", "keras_core.metrics.log_cosh"]
)
def log_cosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.

    Formula:
    ```python
    loss = mean(log(cosh(y_pred - y_true)), axis=-1)
    ```

    Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small
    `x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works
    mostly like the mean squared error, but will not be so strongly affected by
    the occasional wildly incorrect prediction.

    Example:

    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [0., 0.]]
    >>> loss = keras_core.losses.log_cosh(y_true, y_pred)
    0.108

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    log2 = ops.convert_to_tensor(ops.log(2.0), dtype=y_pred.dtype)

    def _logcosh(x):
        return x + ops.softplus(-2.0 * x) - log2

    return ops.mean(_logcosh(y_pred - y_true), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.kl_divergence",
        "keras_core.losses.kl_divergence",
    ]
)
def kl_divergence(y_true, y_pred):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_true * log(y_true / y_pred)
    ```

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.

    Returns:
        KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.kl_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = ops.clip(y_true, 1e-7, 1)
    >>> y_pred = ops.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, y_pred.dtype)
    y_true = ops.clip(y_true, backend.epsilon(), 1)
    y_pred = ops.clip(y_pred, backend.epsilon(), 1)
    return ops.sum(y_true * ops.log(y_true / y_pred), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.poisson",
        "keras_core.losses.poisson",
    ]
)
def poisson(y_true, y_pred):
    """Computes the Poisson loss between y_true and y_pred.

    Formula:

    ```python
    loss = y_pred - y_true * log(y_pred)
    ```

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.poisson(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_pred = y_pred + 1e-7
    >>> assert np.allclose(
    ...     loss, np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
    ...     atol=1e-5)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    epsilon = ops.convert_to_tensor(backend.epsilon())
    return ops.mean(y_pred - y_true * ops.log(y_pred + epsilon), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.categorical_crossentropy",
        "keras_core.losses.categorical_crossentropy",
    ]
)
def categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the categorical crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.

    Returns:
        Categorical crossentropy loss value.

    Example:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras_core.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_crossentropy, expected "
            "y_pred.shape to be (batch_size, num_classes) "
            f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
            "Consider using 'binary_crossentropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )

    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    return ops.categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis
    )


@keras_core_export(
    [
        "keras_core.metrics.categorical_focal_crossentropy",
        "keras_core.losses.categorical_focal_crossentropy",
    ]
)
def categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Computes the categorical focal crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.

    Returns:
        Categorical focal crossentropy loss value.

    Example:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
    >>> loss = keras_core.losses.categorical_focal_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([2.63401289e-04, 6.75912094e-01], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_focal_crossentropy, expected "
            "y_pred.shape to be (batch_size, num_classes) "
            f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
            "Consider using 'binary_crossentropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )

    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    if from_logits:
        y_pred = ops.softmax(y_pred, axis=axis)

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = y_pred / ops.sum(y_pred, axis=axis, keepdims=True)
    output = ops.clip(output, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate cross entropy
    cce = -y_true * ops.log(output)

    # Calculate factors
    modulating_factor = ops.power(1.0 - output, gamma)
    weighting_factor = ops.multiply(modulating_factor, alpha)

    # Apply weighting factor
    focal_cce = ops.multiply(weighting_factor, cce)
    focal_cce = ops.sum(focal_cce, axis=axis)
    return focal_cce


@keras_core_export(
    [
        "keras_core.metrics.sparse_categorical_crossentropy",
        "keras_core.losses.sparse_categorical_crossentropy",
    ]
)
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    """Computes the sparse categorical crossentropy loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.

    Returns:
        Sparse categorical crossentropy loss value.

    Examples:

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras_core.losses.sparse_categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)
    """
    return ops.sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=from_logits,
        axis=axis,
    )


@keras_core_export(
    [
        "keras_core.metrics.binary_crossentropy",
        "keras_core.losses.binary_crossentropy",
    ]
)
def binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the binary crossentropy loss.

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to -1.

    Returns:
        Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras_core.losses.binary_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.916 , 0.714], dtype=float32)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return ops.mean(
        ops.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        axis=axis,
    )


@keras_core_export(
    [
        "keras_core.metrics.binary_focal_crossentropy",
        "keras_core.losses.binary_focal_crossentropy",
    ]
)
def binary_focal_crossentropy(
    y_true,
    y_pred,
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Computes the binary focal crossentropy loss.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output)**gamma` for class 1
    `focal_factor = output**gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
    effect on the binary crossentropy loss.

    If `apply_class_balancing == True`, this function also takes into account a
    weight balancing factor for the binary classes 0 and 1 as follows:

    `weight = alpha` for class 1 (`target == 1`)
    `weight = 1 - alpha` for class 0
    where `alpha` is a float in the range of `[0, 1]`.

    Args:
        y_true: Ground truth values, of shape `(batch_size, d0, .. dN)`.
        y_pred: The predicted values, of shape `(batch_size, d0, .. dN)`.
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Binary focal crossentropy loss value
        with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras_core.losses.binary_focal_crossentropy(
    ...        y_true, y_pred, gamma=2)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.330, 0.206], dtype=float32)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        y_pred = ops.sigmoid(y_pred)

    bce = ops.binary_crossentropy(
        target=y_true,
        output=y_pred,
        from_logits=False,
    )

    # Calculate focal factor
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = ops.power(1.0 - p_t, gamma)

    focal_bce = focal_factor * bce

    if apply_class_balancing:
        weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_bce = weight * focal_bce

    return ops.mean(focal_bce, axis=axis)
