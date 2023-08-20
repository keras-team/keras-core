import logging

import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core.utils import argument_validation
from keras_core.utils import backend_utils
from keras_core.utils import tf_utils
from keras_core.utils.module_utils import tensorflow as tf


def summarize(values, epsilon):
    """Reduce a 1D sequence of values to a summary.

    This algorithm is based on numpy.quantiles but modified to allow for
    intermediate steps between multiple data sets. It first finds the target
    number of bins as the reciprocal of epsilon and then takes the individual
    values spaced at appropriate intervals to arrive at that target.
    The final step is to return the corresponding counts between those values
    If the target num_bins is larger than the size of values, the whole array is
    returned (with weights of 1).

    Args:
        values: 1D `np.ndarray` to be summarized.
        epsilon: A `'float32'` that determines the approximate desired
          precision.

    Returns:
        A 2D `np.ndarray` that is a summary of the inputs. First column is the
        interpolated partition values, the second is the weights (counts).
    """

    values = tf.reshape(values, [-1])
    values = tf.sort(values)
    elements = tf.cast(tf.size(values), tf.float32)
    num_buckets = 1.0 / epsilon
    increment = tf.cast(elements / num_buckets, tf.int32)
    start = increment
    step = tf.maximum(increment, 1)
    boundaries = values[start::step]
    weights = tf.ones_like(boundaries)
    weights = weights * tf.cast(step, tf.float32)
    return tf.stack([boundaries, weights])


def compress(summary, epsilon):
    """Compress a summary to within `epsilon` accuracy.

    The compression step is needed to keep the summary sizes small after
    merging, and also used to return the final target boundaries. It finds the
    new bins based on interpolating cumulative weight percentages from the large
    summary.  Taking the difference of the cumulative weights from the previous
    bin's cumulative weight will give the new weight for that bin.

    Args:
        summary: 2D `np.ndarray` summary to be compressed.
        epsilon: A `'float32'` that determines the approxmiate desired
          precision.

    Returns:
        A 2D `np.ndarray` that is a compressed summary. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    # TODO(b/184863356): remove the numpy escape hatch here.
    return tf.numpy_function(
        lambda s: _compress_summary_numpy(s, epsilon), [summary], tf.float32
    )


def _compress_summary_numpy(summary, epsilon):
    """Compress a summary with numpy."""
    if summary.shape[1] * epsilon < 1:
        return summary

    percents = epsilon + np.arange(0.0, 1.0, epsilon)
    cum_weights = summary[1].cumsum()
    cum_weight_percents = cum_weights / cum_weights[-1]
    new_bins = np.interp(percents, cum_weight_percents, summary[0])
    cum_weights = np.interp(percents, cum_weight_percents, cum_weights)
    new_weights = cum_weights - np.concatenate(
        (np.array([0]), cum_weights[:-1])
    )
    summary = np.stack((new_bins, new_weights))
    return summary.astype(np.float32)


def merge_summaries(prev_summary, next_summary, epsilon):
    """Weighted merge sort of summaries.

    Given two summaries of distinct data, this function merges (and compresses)
    them to stay within `epsilon` error tolerance.

    Args:
        prev_summary: 2D `np.ndarray` summary to be merged with `next_summary`.
        next_summary: 2D `np.ndarray` summary to be merged with `prev_summary`.
        epsilon: A float that determines the approxmiate desired precision.

    Returns:
        A 2-D `np.ndarray` that is a merged summary. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    merged = tf.concat((prev_summary, next_summary), axis=1)
    merged = tf.gather(merged, tf.argsort(merged[0]), axis=1)
    return compress(merged, epsilon)


def get_bin_boundaries(summary, num_bins):
    return compress(summary, 1.0 / num_bins)[0, :-1]


@keras_core_export("keras_core.layers.Discretization")
class Discretization(Layer):
    """A preprocessing layer which buckets continuous features by ranges.

    This layer will place each element of its input data into one of several
    contiguous ranges and output an integer index indicating which range each
    element was placed in.

    **Note:** This layer wraps `tf.keras.layers.Discretization`. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        Any array of dimension 2 or higher.

    Output shape:
        Same as input shape.

    Arguments:
        bin_boundaries: A list of bin boundaries.
            The leftmost and rightmost bins
            will always extend to `-inf` and `inf`,
            so `bin_boundaries=[0., 1., 2.]`
            generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`,
            and `[2., +inf)`.
            If this option is set, `adapt()` should not be called.
        num_bins: The integer number of bins to compute.
            If this option is set,
            `adapt()` should be called to learn the bin boundaries.
        epsilon: Error tolerance, typically a small fraction
            close to zero (e.g. 0.01). Higher values of epsilon increase
            the quantile approximation, and hence result in more
            unequal buckets, but could improve performance
            and resource consumption.
        output_mode: Specification for the output of the layer.
            Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or
            `"count"` configuring the layer as follows:
            - `"int"`: Return the discretized bin indices directly.
            - `"one_hot"`: Encodes each individual element in the
                input into an array the same size as `num_bins`,
                containing a 1 at the input's bin
                index. If the last dimension is size 1, will encode on that
                dimension.  If the last dimension is not size 1,
                will append a new dimension for the encoded output.
            - `"multi_hot"`: Encodes each sample in the input into a
                single array the same size as `num_bins`,
                containing a 1 for each bin index
                index present in the sample.
                Treats the last dimension as the sample
                dimension, if input shape is `(..., sample_length)`,
                output shape will be `(..., num_tokens)`.
            - `"count"`: As `"multi_hot"`, but the int array contains
                a count of the number of times the bin index appeared
                in the sample.
            Defaults to `"int"`.
        sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
            and `"count"` output modes. Only supported with TensorFlow
            backend. If `True`, returns a `SparseTensor` instead of
            a dense `Tensor`. Defaults to `False`.

    Examples:

    Bucketize float values based on provided buckets.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = Discretization(bin_boundaries=[0., 1., 2.])
    >>> layer(input)
    array([[0, 2, 3, 1],
           [1, 3, 2, 1]])

    Bucketize float values based on a number of buckets to compute.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = Discretization(num_bins=4, epsilon=0.01)
    >>> layer.adapt(input)
    >>> layer(input)
    array([[0, 2, 3, 2],
           [1, 3, 3, 1]])
    """

    def __init__(
        self,
        bin_boundaries=None,
        num_bins=None,
        epsilon=0.01,
        output_mode="int",
        sparse=False,
        name=None,
        dtype=None,
        **kwargs,
    ):
        if not tf.available:
            raise ImportError(
                "Layer Discretization requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse` can only be set to True with the "
                "TensorFlow backend."
            )

        if "bins" in kwargs:
            logging.warning(
                "bins is deprecated, "
                "please use bin_boundaries or num_bins instead."
            )
            if isinstance(kwargs["bins"], int) and num_bins is None:
                num_bins = kwargs["bins"]
            elif bin_boundaries is None:
                bin_boundaries = kwargs["bins"]
            del kwargs["bins"]

        if output_mode == "int" and dtype is None:
            dtype = "int64"

        super().__init__(name=name, dtype=dtype)
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse` can only be set to True with the "
                "TensorFlow backend."
            )

        argument_validation.validate_string_arg(
            output_mode,
            allowable_strings=("int", "one_hot", "multi_hot", "count"),
            caller_name=self.__class__.__name__,
            arg_name="output_mode",
        )

        if sparse and output_mode == "int":
            raise ValueError(
                "`sparse` may only be true if `output_mode` is "
                "`'one_hot'`, `'multi_hot'`, or `'count'`. "
                f"Received: sparse={sparse} and "
                f"output_mode={output_mode}"
            )

        if num_bins is not None and num_bins < 0:
            raise ValueError(
                "`num_bins` must be greater than or equal to 0. "
                "You passed `num_bins={}`".format(num_bins)
            )
        if num_bins is not None and bin_boundaries is not None:
            raise ValueError(
                "Both `num_bins` and `bin_boundaries` should not be "
                "set. You passed `num_bins={}` and "
                "`bin_boundaries={}`".format(num_bins, bin_boundaries)
            )

        bin_boundaries = tf_utils.listify_tensors(bin_boundaries)
        self.input_bin_boundaries = bin_boundaries
        self.bin_boundaries = (
            bin_boundaries if bin_boundaries is not None else []
        )
        if self.bin_boundaries:
            self.built = True
            self.summary = None
        else:
            self.summary = self.add_weight(
                name="summary",
                shape=(2, None),
                dtype="float32",
                initializer=lambda shape, dtype: [
                    [],
                    [],
                ],
                trainable=False,
            )
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

        self.num_bins = num_bins
        self.epsilon = epsilon
        self.output_mode = output_mode
        self.sparse = sparse
        self.supports_jit = False

    def build(self, input_shape):
        self.built = True

    # We override this method solely to generate a docstring.
    def adapt(self, data, batch_size=None, steps=None):
        """Computes bin boundaries from quantiles in a input dataset.

        Calling `adapt()` on a `Discretization` layer is an alternative to
        passing in a `bin_boundaries` argument during construction. A
        `Discretization` layer should always be either adapted over a dataset or
        passed `bin_boundaries`.

        During `adapt()`, the layer will estimate the quantile boundaries of the
        input dataset. The number of quantiles can be controlled via the
        `num_bins` argument, and the error tolerance for quantile boundaries can
        be controlled via the `epsilon` argument.

        Arguments:
            data: The data to train on. It can be passed either as a
                batched `tf.data.Dataset`,
                or as a NumPy array.
            batch_size: Integer or `None`.
                Number of samples per state update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of a `tf.data.Dataset`
                (it is expected to be already batched).
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                When training with input tensors such as
                the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
                If `data` is a `tf.data.Dataset`, and `steps` is `None`,
                `adapt()` will run until the input dataset is exhausted.
                When passing an infinitely
                repeating dataset, you must specify the `steps` argument. This
                argument is not supported with array inputs or list inputs.
        """
        # self.layer.adapt(data, batch_size=batch_size, steps=steps)
        print(f"built state {self.built}")
        if self.built:
            self.reset_state()
        else:
            self.build(input_shape=data.shape)

        if isinstance(data, tf.data.Dataset):
            if steps is not None:
                data = data.take(steps)
            for batch in data:
                self.update_state(batch)
        else:
            data = tf_utils.ensure_tensor(data)
            if data.shape.rank == 1:
                # A plain list of strings
                # is treated as as many documents
                data = tf.expand_dims(data, -1)
            self.update_state(data)
        self.finalize_state()

    def update_state(self, data):
        # self.layer.update_state(data)
        if self.input_bin_boundaries is not None:
            raise ValueError(
                "Cannot adapt a Discretization layer that has been initialized "
                "with `bin_boundaries`, use `num_bins` instead. You passed "
                "`bin_boundaries={}`.".format(self.input_bin_boundaries)
            )

        data = backend.convert_to_tensor(data)
        if data.dtype != "float32":
            data = backend.cast(data, "float32")
        summary = summarize(data, self.epsilon)
        self.summary = merge_summaries(summary, self.summary, self.epsilon)

    def finalize_state(self):
        if self.input_bin_boundaries is not None or not self.built:
            return

        # The bucketize op only support list boundaries.
        self.bin_boundaries = tf_utils.listify_tensors(
            get_bin_boundaries(self.summary, self.num_bins)
        )

    def reset_state(self):
        if self.input_bin_boundaries is not None or not self.built:
            return

        self.summary.assign([[], []])

    def compute_output_spec(self, inputs):
        return backend.KerasTensor(shape=inputs.shape, dtype="int32")

    def call(self, inputs):
        def bucketize(inputs):
            return tf.raw_ops.Bucketize(
                input=inputs, boundaries=self.bin_boundaries
            )

        if isinstance(inputs, tf.RaggedTensor):
            indices = tf.ragged.map_flat_values(bucketize, inputs)
        elif isinstance(inputs, tf.SparseTensor):
            indices = tf.SparseTensor(
                indices=tf.identity(inputs.indices),
                values=bucketize(inputs.values),
                dense_shape=tf.identity(inputs.dense_shape),
            )
        else:
            indices = bucketize(inputs)

        outputs = tf_utils.encode_categorical_inputs(
            indices,
            output_mode=self.output_mode,
            depth=len(self.bin_boundaries) + 1,
            sparse=self.sparse,
            dtype=self.compute_dtype,
        )

        if (
            backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            outputs = backend.convert_to_tensor(outputs)
        return outputs

    def get_config(self):
        return {
            "bin_boundaries": self.bin_boundaries,
            "num_bins": self.num_bins,
            "epsilon": self.epsilon,
            "output_mode": self.output_mode,
            "sparse": self.sparse,
            "name": self.name,
            "dtype": self.dtype,
        }
