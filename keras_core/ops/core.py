"""
scatter
scatter_update
slice
slice_update
while_loop
stop_gradient
shape
cast
convert_to_tensor
convert_to_numpy
cond
"""

import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops.operation import Operation
from keras_core.utils import traceback_utils


class Scatter(Operation):
    def call(self, indices, values, shape):
        return backend.core.scatter(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


@keras_core_export("keras_core.ops.scatter")
def scatter(indices, values, shape):
    """Returns a tensor of shape `shape` where `indices` are set to `values`.

    At a high level, this operation does `zeros[indices] = updates` and
    returns the output. It is equivalent to:

    ```python
    zeros = keras_core.ops.zeros(shape)
    output = keras_core.ops.scatter_update(zeros, indices, values)
    ```

    Args:
        indices: A tensor or list/tuple specifying
            indices for the values in `values`.
        updates: A tensor, the values to be set at `indices`.
        shape: Shape of the output tensor.

    Example:

    >>> indices = [[0, 1], [1, 1]]
    >>> values = np.array([1., 1.])
    >>> keras_core.ops.scatter(indices, values, shape=(2, 2))
    array([[0., 1.],
           [0., 1.]])
    """
    if any_symbolic_tensors((indices, values, shape)):
        return Scatter().symbolic_call(indices, values, shape)
    return backend.core.scatter(indices, values, shape)


class ScatterUpdate(Operation):
    def call(self, inputs, indices, updates):
        return backend.core.scatter_update(inputs, indices, updates)

    def compute_output_spec(self, inputs, indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_core_export("keras_core.ops.scatter_update")
def scatter_update(inputs, indices, updates):
    """Update inputs via updates at scattered (sparse) indices.

    At a high level, this operation does `inputs[indices] = updates`.
    Assume `inputs` is a tensor of shape `(D0, D1, ..., Dn)`, there are 2 main
    usages of `scatter_update`.

    1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
        is the number of updates to perform, and `updates` is a 1D tensor of
        shape `(num_updates,)`. For example, if `inputs` is `zeros((4, 4, 4))`,
        and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then
        we can use:

    ```python
    inputs = np.zeros((4, 4, 4))
    indices = [[1, 2, 3], [0, 1, 3]]
    updates = np.array([1., 1.])
    inputs = keras_core.ops.scatter_update(inputs, indices, updates)
    ```

    2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
        is the number of updates to perform, and `k` (`k < n`) is the size of
        each index in `indices`. `updates` is a `n - k`-D tensor of shape
        `(num_updates, inputs.shape[k:])`. For example, if
        `inputs = np.zeros((4, 4, 4))`, and we want to update `inputs[1, 2, :]`
        and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape
        `(num_updates, 2)` (`k = 2`), and `updates` would have shape
        `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:

    ```python
    inputs = np.zeros((4, 4, 4))
    indices = [[1, 2], [2, 3]]
    updates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])
    inputs = keras_core.ops.scatter_update(inputs, indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        indices: A tensor or list/tuple of shape `(N, inputs.ndim)`, specifying
            indices to update. `N` is the number of indices to update, must be
            equal to the first dimension of `updates`.
        updates: A tensor, the new values to be put to `inputs` at `indices`.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, indices, updates)):
        return ScatterUpdate().symbolic_call(inputs, indices, updates)
    return backend.core.scatter_update(inputs, indices, updates)


class Slice(Operation):
    def call(self, inputs, start_indices, shape):
        return backend.core.slice(inputs, start_indices, shape)

    def compute_output_spec(self, inputs, start_indices, shape):
        return KerasTensor(shape, dtype=inputs.dtype)


@keras_core_export("keras_core.ops.slice")
def slice(inputs, start_indices, shape):
    """Return a slice of an input tensor.

    At a high level, this operation is an explicit replacement for array slicing
    e.g. `inputs[start_indices: start_indices + shape]`.
    Unlike slicing via brackets, this operation will accept tensor start
    indices on all backends, which is useful when indices dynamically computed
    via other tensor operations.

    ```python
    inputs = np.zeros((5, 5))
    start_indices = np.array([3, 3])
    shape = np.array([2, 2])
    inputs = keras_core.ops.slice(inputs, start_indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying
            the starting indices for updating.
        shape: The full shape of the returned slice.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices, shape)):
        return Slice().symbolic_call(inputs, start_indices, shape)
    return backend.core.slice(inputs, start_indices, shape)


class SliceUpdate(Operation):
    def call(self, inputs, start_indices, updates):
        return backend.core.slice_update(inputs, start_indices, updates)

    def compute_output_spec(self, inputs, start_indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_core_export("keras_core.ops.slice_update")
def slice_update(inputs, start_indices, updates):
    """Update an input by slicing in a tensor of updated values.

    At a high level, this operation does
    `inputs[start_indices: start_indices + updates.shape] = updates`.
    Assume inputs is a tensor of shape `(D0, D1, ..., Dn)`,
    `start_indices` must be a list/tuple of n integers, specifying the starting
    indices. `updates` must have the same rank as `inputs`, and the size of each
    dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
    inputs `inputs = np.zeros((5, 5))`, and we want to update the intersection
    of last 2 rows and last 2 columns as 1, i.e.,
    `inputs[3:, 3:] = np.ones((2, 2))`, then we can use the code below:

    ```python
    inputs = np.zeros((5, 5))
    start_indices = [3, 3]
    updates = np.ones((2, 2))
    inputs = keras_core.ops.slice_update(inputs, start_indices, updates)
    ```

    Args:
        inputs: A tensor, the tensor to be updated.
        start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying
            the starting indices for updating.
        updates: A tensor, the new values to be put to `inputs` at `indices`.
            `updates` must have the same rank as `inputs`.

    Returns:
        A tensor, has the same shape and dtype as `inputs`.
    """
    if any_symbolic_tensors((inputs, start_indices, updates)):
        return SliceUpdate().symbolic_call(inputs, start_indices, updates)
    return backend.core.slice_update(inputs, start_indices, updates)


class WhileLoop(Operation):
    def __init__(self, cond, body, maximum_iterations):
        super().__init__()
        self.cond = cond
        self.body = body
        self.maximum_iterations = maximum_iterations

    def call(self, loop_vars):
        return backend.core.while_loop(
            self.cond,
            self.body,
            loop_vars,
            maximum_iterations=self.maximum_iterations,
        )

    def compute_output_spec(self, loop_vars):
        return [KerasTensor(v.shape, dtype=v.dtype) for v in loop_vars]


@keras_core_export("keras_core.ops.while_loop")
def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    """While loop implementation.

    Args:
        cond: A callable that represents the termination condition of the loop.
            Must have the same number of args as `loop_vars`, and return a bool.
        body: A callable that represents the loop body. Must have the same
            number of args as `loop_vars`, and return a list/tuple of the same
            length, shape and dtype as `loop_vars`.
        loop_vars: A list/tuple of tensors, the loop variables.
        maximum_iterations: Optional maximum number of iterations of the while
            loop to run. If provided, the `cond` output is AND-ed with an
            additional condition ensuring the number of iterations executed is
            no greater than `maximum_iterations`.

    Returns:
        A list/tuple of tensors, has the same shape and dtype as `inputs`.

    Examples:

    >>> i = 0
    >>> cond = lambda i: i < 10
    >>> body = lambda i: i + 1
    >>> keras_core.ops.while_loop(cond, body, [i])[0]
    10
    """
    return backend.core.while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=maximum_iterations,
    )


class StopGradient(Operation):
    def __init__(self):
        super().__init__()

    def call(self, variable):
        return backend.core.stop_gradient(variable)

    def compute_output_spec(self, variable):
        return KerasTensor(variable.shape, dtype=variable.dtype)


@keras_core_export("keras_core.ops.stop_gradient")
def stop_gradient(variable):
    """Stops gradient computation.

    Args:
        variable: A tensor variable for which the gradient
            computation is to be disabled.

    Returns:
        The variable with gradient computation disabled.

    Examples:

    >>> var = keras_core.backend.convert_to_tensor(
    ...     [1., 2., 3.],
    ...     dtype="float32"
    ... )
    >>> var = keras_core.ops.stop_gradient(var)
    """
    return backend.core.stop_gradient(variable)


class ForiLoop(Operation):
    def __init__(self, lower, upper, body_fun):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.body_fun = body_fun

    def call(self, init_val):
        return backend.core.fori_loop(
            self.lower,
            self.upper,
            self.body_fun,
            init_val,
        )

    def compute_output_spec(self, init_val):
        return KerasTensor(init_val.shape, dtype=init_val.dtype)


@keras_core_export("keras_core.ops.fori_loop")
def fori_loop(lower, upper, body_fun, init_val):
    """For loop implementation.

    Args:
        lower: The initial value of the loop variable.
        upper: The upper bound of the loop variable.
        body_fun: A callable that represents the loop body. Must take two
            arguments: the loop variable and the loop state. The loop state
            should be updated and returned by this function.
        init_val: The initial value of the loop state.

    Returns:
        The final state after the loop.

    Example:

    >>> lower = 0
    >>> upper = 10
    >>> body_fun = lambda i, s: (i + 1, s + i)
    >>> init_val = 0
    >>> keras_core.ops.fori_loop(lower, upper, body_fun, init_val)
    45
    """
    if any_symbolic_tensors((lower, upper, init_val)):
        return ForiLoop(lower, upper, body_fun).symbolic_call(init_val)
    return backend.core.fori_loop(lower, upper, body_fun, init_val)


class Unstack(Operation):
    def __init__(self, num=None, axis=0):
        super().__init__()
        self.num = num
        self.axis = axis

    def call(self, x):
        return backend.core.unstack(x, self.num, self.axis)

    def compute_output_spec(self, x):
        axis = self.axis
        if axis < 0:
            axis = len(x.shape) + axis
        output_shapes = x.shape[:axis] + x.shape[axis + 1 :]
        num = self.num
        if num is None:
            num = x.shape[axis]
        if num is None:
            raise ValueError(
                "Cannot infer argument `num` from shape "
                f"{x.shape}. Either provide a tensor with a "
                "concrete shape in the `axis` dimension or "
                "explicitly pass the `num` argument."
            )
        output = [
            KerasTensor(shape=output_shapes, dtype=x.dtype) for _ in range(num)
        ]
        return output


@keras_core_export("keras_core.ops.unstack")
def unstack(x, num=None, axis=0):
    """Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

    Args:
        x: The input tensor.
        num: The length of the dimension axis. Automatically inferred
            if `None`.
        axis: The axis along which to unpack.

    Returns:
        A list of tensors unpacked along the given axis.

    Example:

    >>> x = keras_core.ops.array([[1, 2], [3, 4]])
    >>> keras_core.ops.unstack(x, axis=0)
    [array([1, 2]), array([3, 4])]
    """
    if any_symbolic_tensors((x,)):
        return Unstack(num, axis).symbolic_call(x)
    return backend.core.unstack(x, num=num, axis=axis)


@keras_core_export("keras_core.ops.shape")
def shape(x):
    """Gets the shape of the tensor input.

    Note: On the tensorflow backend, when `x` is a `tf.Tensor` with dynamic
    shape, dimensions which are dynamic in the context of a compiled function
    will have a `tf.Tensor` value instead of a static integer value.

    Args:
        x: A tensor. This function will try to access the `shape` attribute of
            the input tensor.

    Returns:
        A tuple of integers or None values, indicating the shape of the input
            tensor.

    Example:

    >>> x = keras_core.zeros((8, 12))
    >>> keras_core.ops.shape(x)
    (8, 12)
    """
    if any_symbolic_tensors((x,)):
        return x.shape
    return backend.core.shape(x)


class Cast(Operation):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = backend.standardize_dtype(dtype)

    def call(self, x):
        return backend.core.cast(x, self.dtype)

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)


@keras_core_export("keras_core.ops.cast")
def cast(x, dtype):
    """Cast a tensor to the desired dtype.

    Args:
        x: A tensor or variable.
        dtype: The target type.

    Returns:
        A tensor of the specified `dtype`.

    Example:

    >>> x = keras_core.ops.arange(4)
    >>> x = keras_core.ops.cast(x, dtype="float16")
    """
    dtype = backend.standardize_dtype(dtype)

    if any_symbolic_tensors((x,)):
        return Cast(dtype=dtype)(x)
    return backend.core.cast(x, dtype)


@keras_core_export("keras_core.ops.convert_to_tensor")
def convert_to_tensor(x, dtype=None):
    """Convert a NumPy array to a tensor.

    Args:
        x: A NumPy array.
        dtype: The target type.

    Returns:
        A tensor of the specified `dtype`.

    Example:

    >>> x = np.array([1, 2, 3])
    >>> y = keras_core.ops.convert_to_tensor(x)
    """
    return backend.convert_to_tensor(x, dtype=dtype)


@keras_core_export("keras_core.ops.convert_to_numpy")
def convert_to_numpy(x):
    """Convert a tensor to a NumPy array.

    Args:
        x: A tensor.

    Returns:
        A NumPy array.
    """
    if any_symbolic_tensors((x,)):
        # This will raise a `ValueError` defined in the `KerasTensor` class.
        # We trigger it rather than duplicate it here.
        return np.array(x)
    return backend.convert_to_numpy(x)


class Cond(Operation):
    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        def call_fn(*args, **kwargs):
            if not any_symbolic_tensors(args, kwargs):
                try:
                    return self.call(*args, **kwargs)
                except (TypeError, ValueError):
                    # fallback on symbolic case
                    pass
            return self.symbolic_call(*args, **kwargs)

        if traceback_utils.is_traceback_filtering_enabled():
            # Wrap self.call to provide helpful info in case of exception
            call_fn = traceback_utils.inject_argument_info_in_traceback(
                call_fn,
                object_name=(f"{self.__class__.__name__}.call()"),
            )
            return call_fn(*args, **kwargs)

        # Plain flow.
        return call_fn(*args, **kwargs)

    def call(self, pred, true_fn, false_fn):
        return backend.core.cond(pred, true_fn, false_fn)

    def compute_output_spec(self, pred, true_fn, false_fn):
        def call_fn(fn):
            return fn()

        true_fn_spec = backend.compute_output_spec(call_fn, true_fn)
        false_fn_spec = backend.compute_output_spec(call_fn, false_fn)
        if not self._check_output_spec(true_fn_spec, false_fn_spec):
            raise ValueError(
                "`true_fn` and `false_fn` should return outputs "
                "of the same kind (struct, dtype and shape). "
                f"Got {true_fn_spec} and {false_fn_spec} instead."
            )
        return true_fn_spec

    def _check_output_spec(self, true_fn_spec, false_fn_spec):
        if isinstance(true_fn_spec, dict):
            if not isinstance(false_fn_spec, dict):
                return False
            if true_fn_spec.keys() != false_fn_spec.keys():
                return False
            if any(
                (not self._check_output_spec(true_fn_spec[k], false_fn_spec[k]))
                for k in true_fn_spec.keys()
            ):
                return False
        elif isinstance(true_fn_spec, list):
            if not isinstance(false_fn_spec, list):
                return False
            if len(true_fn_spec) != len(false_fn_spec):
                return False
            if any(
                (not self._check_output_spec(ti, fi))
                for ti, fi in zip(true_fn_spec, false_fn_spec)
            ):
                return False
        else:
            if true_fn_spec.dtype != false_fn_spec.dtype:
                return False
            if true_fn_spec.shape != false_fn_spec.shape:
                return False

        return True


@keras_core_export("keras_core.ops.cond")
def cond(pred, true_fn, false_fn):
    """Conditionally applies `true_fn` or `false_fn`.

    Args:
        pred: Boolean scalar type
        true_fn: Callable returning the output for the `pred == True` case.
        false_fn: Callable returning the output for the `pred == False` case.

    Returns:
        The output of either `true_fn` or `false_fn` depending on pred.
    """
    return Cond()(pred, true_fn, false_fn)
