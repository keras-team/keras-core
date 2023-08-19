"""
MANIFEST:

abs
absolute
add
all
amax
amin
append
arange
arccos
arccosh
arcsin
arcsinh
arctan
arctan2
arctanh
argmax
argmin
argsort
array
average
bincount
broadcast_to
ceil
clip
concatenate
conj
conjugate
copy
cos
cosh
count_nonzero
cross
cumprod
cumsum
diag
diagonal
diff
digitize
divide
dot
dtype
einsum
empty
equal
exp
expand_dims
expm1
eye
flip
floor
full
full_like
greater
greater_equal
hstack
identity
imag
interp
isclose
isfinite
isinf
isnan
less
less_equal
linspace
log
log10
log1p
log2
logaddexp
logical_and
logical_not
logical_or
logspace
matmul
max
maximum
mean
median
meshgrid
mgrid
min
minimum
mod
moveaxis
multiply
nan_to_num
ndim
nonzero
not_equal
ones
ones_like
outer
pad
percentile
power
prod
ravel
real
reciprocal
repeat
reshape
roll
round
sign
sin
sinh
size
sort
split
sqrt
square
squeeze
stack
std
subtract
sum
swapaxes
take
take_along_axis
tan
tanh
tensordot
tile
trace
transpose
tri
tril
triu
true_divide
vdot
vstack
where
zeros
zeros_like


"""
import re

import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops import operation_utils
from keras_core.ops.operation import Operation
from keras_core.ops.operation_utils import reduce_shape


def broadcast_shapes(shape1, shape2):
    """Broadcast input shapes to a unified shape.

    Convert to list for mutability.

    Args:
        shape1: A tuple or list of integers.
        shape2: A tuple or list of integers.

    Returns:
        output_shape (list of integers or `None`): The broadcasted shape.

    Example:
    >>> broadcast_shapes((5, 3), (1, 3))
    [5, 3]
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = shape1
    origin_shape2 = shape2

    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = None if shape2[i] == 1 else shape2[i]
        else:
            if shape2[i] == 1 or shape2[i] is None or shape2[i] == shape1[i]:
                output_shape[i] = shape1[i]
            else:
                raise ValueError(
                    "Cannot broadcast shape, the failure dim has value "
                    f"{shape1[i]}, which cannot be broadcasted to {shape2[i]}. "
                    f"Input shapes are: {origin_shape1} and {origin_shape2}."
                )

    return output_shape


def shape_equal(shape1, shape2, axis=None, allow_none=True):
    """Check if two shapes are equal

    Args:
        shape1: A list or tuple of integers for first shape to be compared.
        shape2: A list or tuple of integers for second shape to be compared.
        axis: An integer, list, or tuple of integers (optional):
            Axes to ignore during comparison.Default is None.
        allow_none (bool, optional): If True, allows None in a shape to match
            any value in the corresponding position of the other shape.
            Default is True.

    Returns:
        bool: True if shapes are considered equal based on the criteria,
        False otherwise.

    Examples:

        >>> shape_equal((32, 64, 128), (32, 64, 128))
        True
        >>> shape_equal((32, 64, 128), (32, 64, 127))
        False
        >>> shape_equal((32, 64, None), (32, 64, 128), allow_none=True)
        True
        >>> shape_equal((32, 64, None), (32, 64, 128), allow_none=False)
        False
        >>> shape_equal((32, 64, 128), (32, 63, 128), axis=1)
        True
        >>> shape_equal((32, 64, 128), (32, 63, 127), axis=(1, 2))
        True
        >>> shape_equal((32, 64, 128), (32, 63, 127), axis=1)
        False
        >>> shape_equal((32, 64), (32, 64, 128))
        False
    """
    if len(shape1) != len(shape2):
        return False

    shape1 = list(shape1)
    shape2 = list(shape2)

    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        for ax in axis:
            shape1[ax] = -1
            shape2[ax] = -1

    if allow_none:
        for i in range(len(shape1)):
            if shape1[i] is None:
                shape1[i] = shape2[i]
            if shape2[i] is None:
                shape2[i] = shape1[i]

    return shape1 == shape2


class Absolute(Operation):
    def call(self, x):
        return backend.numpy.absolute(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.absolute", "keras_core.ops.numpy.absolute"])
def absolute(x):
    """Compute the absolute value element-wise.

    `keras_core.ops.abs` is a shorthand for this function.

    Args:
        x: Input tensor.

    Returns:
        An array containing the absolute value of each element in `x`.

    Example:
    >>> x = keras_core.ops.convert_to_tensor([-1.2, 1.2])
    >>> keras_core.ops.absolute(x)
    array([1.2 1.2], shape=(2,), dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.numpy.absolute(x)


class Abs(Absolute):
    pass


@keras_core_export(["keras_core.ops.abs", "keras_core.ops.numpy.abs"])
def abs(x):
    """Shorthand for `keras_core.ops.absolute`."""
    return absolute(x)


class Add(Operation):
    def call(self, x1, x2):
        return backend.numpy.add(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.add", "keras_core.ops.numpy.add"])
def add(x1, x2):
    """Add arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        The tensor containing the element-wise sum of `x1` and `x2`.

    Examples:
    >>> x1 = keras_core.ops.convert_to_tensor([1, 4])
    >>> x2 = keras_core.ops.convert_to_tensor([5, 6])
    >>> keras_core.ops.add(x1, x2)
    array([ 6 10], shape=(2,), dtype=int32)

    `keras_core.ops.add` also broadcasts shapes:
    >>> x1 = keras_core.ops.convert_to_tensor(
    ...     [[5, 4],
    ...      [5, 6]]
    ... )
    >>> x2 = keras_core.ops.convert_to_tensor([5, 6])
    >>> keras_core.ops.add(x1, x2)
    array([[10 10]
           [10 12]], shape=(2, 2), dtype=int32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Add().symbolic_call(x1, x2)
    return backend.numpy.add(x1, x2)


class All(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.all(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(
                x.shape,
                axis=self.axis,
                keepdims=self.keepdims,
            ),
            dtype="bool",
        )


@keras_core_export(["keras_core.ops.all", "keras_core.ops.numpy.all"])
def all(x, axis=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to `True`.

    Args:
        x: Input tensor.
        axis: An integer or tuple of integers that represent the axis along
            which a logical AND reduction is performed. The default
            (`axis=None`) is to perform a logical AND over all the dimensions
            of the input array. `axis` may be negative, in which case it counts
            for the last to the first axis.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the input array. Default is `False`.

    Returns:
        The tensor containing the logical AND reduction over the `axis`.

    Examples:
    >>> x = keras_core.ops.convert_to_tensor([True, False])
    >>> keras_core.ops.all(x)
    array(False, shape=(), dtype=bool)

    >>> x = keras_core.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras_core.ops.all(x, axis=0)
    array([ True False], shape=(2,), dtype=bool)

    `keepdims=True` outputs a tensor with dimensions reduced to one.
    >>> x = keras_core.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras_core.ops.all(x, keepdims=True)
    array([[False]], shape=(1, 1), dtype=bool)
    """
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.all(x, axis=axis, keepdims=keepdims)


class Any(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.any(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(
                x.shape,
                axis=self.axis,
                keepdims=self.keepdims,
            ),
            dtype="bool",
        )


@keras_core_export(["keras_core.ops.any", "keras_core.ops.numpy.any"])
def any(x, axis=None, keepdims=False):
    """Test whether any array element along a given axis evaluates to `True`.

    Args:
        x: Input tensor.
        axis: An integer or tuple of integers that represent the axis along
            which a logical OR reduction is performed. The default
            (`axis=None`) is to perform a logical OR over all the dimensions
            of the input array. `axis` may be negative, in which case it counts
            for the last to the first axis.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the input array. Default is `False`.

    Returns:
        The tensor containing the logical OR reduction over the `axis`.

    Examples:
    >>> x = keras_core.ops.convert_to_tensor([True, False])
    >>> keras_core.ops.any(x)
    array(True, shape=(), dtype=bool)

    >>> x = keras_core.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras_core.ops.any(x, axis=0)
    array([ True  True], shape=(2,), dtype=bool)

    `keepdims=True` outputs a tensor with dimensions reduced to one.
    >>> x = keras_core.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras_core.ops.all(x, keepdims=True)
    array([[False]], shape=(1, 1), dtype=bool)
    """
    if any_symbolic_tensors((x,)):
        return Any(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.any(x, axis=axis, keepdims=keepdims)


class Amax(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.amax(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.amax", "keras_core.ops.numpy.amax"])
def amax(x, axis=None, keepdims=False):
    """Returns the maximum of an array or maximum value along an axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute the maximum.
            By default (`axis=None`), find the maximum value in all the
            dimensions of the input array.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions that are broadcast to the size of the original
            input tensor. Defaults to `False`.

    Returns:
        An array with the maximum value. If `axis=None`, the result is a scalar
        value representing the maximum element in the entire array. If `axis` is
        given, the result is an array with the maximum values along
        the specified axis.

    Examples:
    >>> x = keras_core.ops.convert_to_tensor([[1, 3, 5], [2, 3, 6]])
    >>> keras_core.ops.amax(x)
    array(6, dtype=int32)

    >>> x = keras_core.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]])
    >>> keras_core.ops.amax(x, axis=0)
    array([1, 6, 8], dtype=int32)

    >>> x = keras_core.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]])
    >>> keras_core.ops.amax(x, axis=1, keepdims=True)
    array([[8], [5]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Amax(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.amax(x, axis=axis, keepdims=keepdims)


class Amin(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.amin(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.amin", "keras_core.ops.numpy.amin"])
def amin(x, axis=None, keepdims=False):
    """Returns the minimum of an array or minimum value along an axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute the minimum.
            By default (`axis=None`), find the minimum value in all the
            dimensions of the input array.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions that are broadcast to the size of the original
            input tensor. Defaults to `False`.

    Returns:
        An array with the minimum value. If `axis=None`, the result is a scalar
        value representing the minimum element in the entire array. If `axis` is
        given, the result is an array with the minimum values along
        the specified axis.

    Examples:
    >>> x = keras_core.ops.convert_to_tensor([1, 3, 5, 2, 3, 6])
    >>> keras_core.ops.amin(x)
    array(1, dtype=int32)

    >>> x = keras_core.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]])
    >>> keras_core.ops.amin(x, axis=0)
    array([1,5,3], dtype=int32)

    >>> x = keras_core.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]])
    >>> keras_core.ops.amin(x, axis=1, keepdims=True)
    array([[1],[3]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Amin(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.amin(x, axis=axis, keepdims=keepdims)


class Append(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x1, x2):
        return backend.numpy.append(x1, x2, axis=self.axis)

    def compute_output_spec(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        if self.axis is None:
            if None in x1_shape or None in x2_shape:
                output_shape = [None]
            else:
                output_shape = [int(np.prod(x1_shape) + np.prod(x2_shape))]
            return KerasTensor(output_shape, dtype=x1.dtype)

        if not shape_equal(x1_shape, x2_shape, [self.axis]):
            raise ValueError(
                "`append` requires inputs to have the same shape except the "
                f"`axis={self.axis}`, but received shape {x1_shape} and "
                f"{x2_shape}."
            )

        output_shape = list(x1_shape)
        output_shape[self.axis] = x1_shape[self.axis] + x2_shape[self.axis]
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.append", "keras_core.ops.numpy.append"])
def append(
    x1,
    x2,
    axis=None,
):
    """Append tensor `x2` to the end of tensor `x1`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        axis: Axis along which tensor `x2` is appended to tensor `x1`.
            If `None`, both tensors are flattened before use.

    Returns:
        A tensor with the values of `x2` appended to `x1`.

    Examples:
    >>> x1 = keras_core.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras_core.ops.convert_to_tensor([[4, 5, 6], [7, 8, 9]])
    >>> keras_core.ops.append(x1, x2)
    array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

    When `axis` is specified, `x1` and `x2` must have compatible shapes.
    >>> x1 = keras_core.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> x2 = keras_core.ops.convert_to_tensor([[7, 8, 9]])
    >>> keras_core.ops.append(x1, x2, axis=0)
    array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=int32)
    >>> x3 = keras_core.ops.convert_to_tensor([7, 8, 9])
    >>> keras_core.ops.append(x1, x3, axis=0)
    Traceback (most recent call last):
        ...
    TypeError: Cannot concatenate arrays with different numbers of
    dimensions: got (2, 3), (3,).
    """
    if any_symbolic_tensors((x1, x2)):
        return Append(axis=axis).symbolic_call(x1, x2)
    return backend.numpy.append(x1, x2, axis=axis)


class Arange(Operation):
    def call(self, start, stop=None, step=1, dtype=None):
        return backend.numpy.arange(start, stop, step=step, dtype=dtype)

    def compute_output_spec(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            start, stop = 0, start
        output_shape = [np.ceil((stop - start) / step).astype(int)]
        return KerasTensor(output_shape, dtype=dtype)


@keras_core_export(["keras_core.ops.arange", "keras_core.ops.numpy.arange"])
def arange(start, stop=None, step=1, dtype=None):
    """Return evenly spaced values within a given interval.

    `arange` can be called with a varying number of positional arguments:
    * `arange(stop)`: Values are generated within the half-open interval
        `[0, stop)` (in other words, the interval including start but excluding
        stop).
    * `arange(start, stop)`: Values are generated within the half-open interval
        `[start, stop)`.
    * `arange(start, stop, step)`: Values are generated within the half-open
        interval `[start, stop)`, with spacing between values given by step.

    Args:
        start: Integer or real, representing the start of the interval. The
            interval includes this value.
        stop: Integer or real, representing the end of the interval. The
            interval does not include this value, except in some cases where
            `step` is not an integer and floating point round-off affects the
            lenght of `out`. Defaults to `None`.
        step: Integer or real, represent the spacing between values. For any
            output `out`, this is the distance between two adjacent values,
            `out[i+1] - out[i]`. The default step size is 1. If `step` is
            specified as a position argument, `start` must also be given.
        dtype: The type of the output array. If `dtype` is not given, infer the
            data type from the other input arguments.

    Returns:
        Tensor of evenly spaced values.
        For floating point arguments, the length of the result is
        `ceil((stop - start)/step)`. Because of floating point overflow, this
        rule may result in the last element of out being greater than stop.

    Examples:
    >>> keras_core.ops.arange(3)
    array([0, 1, 2], dtype=int32)

    >>> keras_core.ops.arange(3.0)
    array([0., 1., 2.], dtype=float32)

    >>> keras_core.ops.arange(3, 7)
    array([3, 4, 5, 6], dtype=int32)

    >>> keras_core.ops.arange(3, 7, 2)
    array([3, 5], dtype=int32)
    """
    return backend.numpy.arange(start, stop, step=step, dtype=dtype)


class Arccos(Operation):
    def call(self, x):
        return backend.numpy.arccos(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.arccos", "keras_core.ops.numpy.arccos"])
def arccos(x):
    """Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if `y = cos(x)`, then `x = arccos(y)`.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the angle of the ray intersecting the unit circle at the given
        x-coordinate in radians `[0, pi]`.

    Example:
    >>> x = keras_core.ops.convert_to_tensor([1, -1])
    >>> keras_core.ops.arccos(x)
    array([0.0, 3.1415927], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arccos().symbolic_call(x)
    return backend.numpy.arccos(x)


class Arccosh(Operation):
    def call(self, x):
        return backend.numpy.arccosh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arccosh(x):
    """Inverse hyperbolic cosine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as x.

    Example:
    >>> x = keras_core.ops.convert_to_tensor([10, 100])
    >>> keras_core.ops.arccosh(x)
    array([2.993223, 5.298292], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arccosh().symbolic_call(x)
    return backend.numpy.arccosh(x)


class Arcsin(Operation):
    def call(self, x):
        return backend.numpy.arcsin(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.arcsin", "keras_core.ops.numpy.arcsin"])
def arcsin(x):
    """Inverse sine, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the inverse sine of each element in `x`, in radians and in
        the closed interval `[-pi/2, pi/2]`.

    Example:
    >>> x = keras_core.ops.convert_to_tensor([1, -1, 0])
    >>> keras_core.ops.arcsin(x)
    array([ 1.5707964, -1.5707964,  0.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arcsin().symbolic_call(x)
    return backend.numpy.arcsin(x)


class Arcsinh(Operation):
    def call(self, x):
        return backend.numpy.arcsinh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.arcsinh", "keras_core.ops.numpy.arcsinh"])
def arcsinh(x):
    """Inverse hyperbolic sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.

    Example:
    >>> x = keras_core.ops.convert_to_tensor([1, -1, 0])
    >>> keras_core.ops.arcsinh(x)
    array([0.88137364, -0.88137364, 0.0], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arcsinh().symbolic_call(x)
    return backend.numpy.arcsinh(x)


class Arctan(Operation):
    def call(self, x):
        return backend.numpy.arctan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.arctan", "keras_core.ops.numpy.arctan"])
def arctan(x):
    """Trigonometric inverse tangent, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the inverse tangent of each element in `x`, in the interval
        `[-pi/2, pi/2]`.

    Example:
    >>> x = keras_core.ops.convert_to_tensor([0, 1])
    >>> keras_core.ops.arctan(x)
    array([0., 0.7853982], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arctan().symbolic_call(x)
    return backend.numpy.arctan(x)


class Arctan2(Operation):
    def call(self, x1, x2):
        return backend.numpy.arctan2(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        outputs_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(outputs_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.arctan2", "keras_core.ops.numpy.arctan2"])
def arctan2(x1, x2):
    """Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that `arctan2(x1, x2)` is the
    signed angle in radians between the ray ending at the origin and passing
    through the point `(1, 0)`, and the ray ending at the origin and passing
    through the point `(x2, x1)`. (Note the role reversal: the "y-coordinate"
    is the first function parameter, the "x-coordinate" is the second.) By IEEE
    convention, this function is defined for `x2 = +/-0` and for either or both
    of `x1` and `x2` `= +/-inf`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Tensor of angles in radians, in the range `[-pi, pi]`.

    Examples:
    Consider four points in different quadrants:
    >>> x = keras_core.ops.convert_to_tensor([-1, +1, +1, -1])
    >>> y = keras_core.ops.convert_to_tensor([-1, -1, +1, +1])
    >>> keras_core.ops.arctan2(y, x) * 180 / numpy.pi
    array([-135., -45., 45., 135.], dtype=float32)

    Note the order of the parameters. `arctan2` is defined also when x2=0 and
    at several other points, obtaining values in the range `[-pi, pi]`:
    >>> keras_core.ops.arctan2(
    ...     keras_core.ops.array([1., -1.]),
    ...     keras_core.ops.array([0., 0.]),
    ... )
    array([ 1.5707964, -1.5707964], dtype=float32)
    >>> keras_core.ops.arctan2(
    ...     keras_core.ops.array([0., 0., numpy.inf]),
    ...     keras_core.ops.array([+0., -0., numpy.inf]),
    ... )
    array([0., 3.1415925, 0.7853982], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Arctan2().symbolic_call(x1, x2)
    return backend.numpy.arctan2(x1, x2)


class Arctanh(Operation):
    def call(self, x):
        return backend.numpy.arctanh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.arctanh", "keras_core.ops.numpy.arctanh"])
def arctanh(x):
    """Inverse hyperbolic tangent, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Arctanh().symbolic_call(x)
    return backend.numpy.arctanh(x)


class Argmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.argmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


@keras_core_export(["keras_core.ops.argmax", "keras_core.ops.numpy.argmax"])
def argmax(x, axis=None):
    """Returns the indices of the maximum values along an axis.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed.

    Example:
    >>> x = keras_core.ops.arange(6).reshape(2, 3) + 10
    >>> x
    array([[10, 11, 12],
           [13, 14, 15]], dtype=int32)
    >>> keras_core.ops.argmax(x)
    array(5, dtype=int32)
    >>> keras_core.ops.argmax(x, axis=0)
    array([1, 1, 1], dtype=int32)
    >>> keras_core.ops.argmax(x, axis=1)
    array([2, 2], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argmax(axis=axis).symbolic_call(x)
    return backend.numpy.argmax(x, axis=axis)


class Argmin(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.argmin(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


@keras_core_export(["keras_core.ops.argmin", "keras_core.ops.numpy.argmin"])
def argmin(x, axis=None):
    """Returns the indices of the minium values along an axis.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed.

    Example:
    >>> x = keras_core.ops.arange(6).reshape(2, 3) + 10
    >>> x
    array([[10, 11, 12],
           [13, 14, 15]], dtype=int32)
    >>> keras_core.ops.argmin(x)
    array(0, dtype=int32)
    >>> keras_core.ops.argmin(x, axis=0)
    array([0, 0, 0], dtype=int32)
    >>> keras_core.ops.argmin(x, axis=1)
    array([0, 0], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argmin(axis=axis).symbolic_call(x)
    return backend.numpy.argmin(x, axis=axis)


class Argsort(Operation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.argsort(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([int(np.prod(x.shape))], dtype="int32")
        return KerasTensor(x.shape, dtype="int32")


@keras_core_export(["keras_core.ops.argsort", "keras_core.ops.numpy.argsort"])
def argsort(x, axis=-1):
    """Returns the indices that would sort a tensor.

    Args:
        x: Input tensor.
        axis: Axis along which to sort. Default is `-1` (the last axis). If
            `None`, the flattened tensor is used.

    Returns:
        Tensor of indices that sort `x` along the specified `axis`.

    Examples:
    One dimensional array:
    >>> x = keras_core.ops.array([3, 1, 2])
    >>> keras_core.ops.argsort(x)
    array([1, 2, 0], dtype=int32)

    Two-dimensional array:
    >>> x = keras_core.ops.array([[0, 3], [3, 2], [4, 5]])
    >>> x
    array([[0, 3],
           [3, 2],
           [4, 5]], dtype=int32)
    >>> keras_core.ops.argsort(x, axis=0)
    array([[0, 1],
           [1, 0],
           [2, 2]], dtype=int32)
    >>> keras_core.ops.argsort(x, axis=1)
    array([[0, 1],
           [1, 0],
           [0, 1]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argsort(axis=axis).symbolic_call(x)
    return backend.numpy.argsort(x, axis=axis)


class Array(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.array(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


@keras_core_export(["keras_core.ops.array", "keras_core.ops.numpy.array"])
def array(x, dtype=None):
    """Create a tensor.

    Args:
        x: Input tensor.
        dtype: The desired data-type for the tensor.

    Returns:
        A tensor.

    Examples:
    >>> keras_core.ops.array([1, 2, 3])
    array([1, 2, 3], dtype=int32)

    >>> keras_core.ops.array([1, 2, 3], dtype="float32")
    array([1., 2., 3.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Array().symbolic_call(x, dtype=dtype)
    return backend.numpy.array(x, dtype=dtype)


class Average(Operation):
    def __init__(self, axis=None):
        super().__init__()
        # np.average() does not support axis as tuple as declared by the
        # docstring, it only supports int or None.
        self.axis = axis

    def call(self, x, weights=None):
        return backend.numpy.average(x, weights=weights, axis=self.axis)

    def compute_output_spec(self, x, weights=None):
        if weights is not None:
            shape_match = shape_equal(x.shape, weights.shape, allow_none=True)
            if self.axis is not None:
                shape_match_on_axis = shape_equal(
                    [x.shape[self.axis]], weights.shape, allow_none=True
                )
        if self.axis is None:
            if weights is None or shape_match:
                return KerasTensor(
                    [],
                    dtype=x.dtype,
                )
            else:
                raise ValueError(
                    "`weights` must have the same shape as `x` when "
                    f"`axis=None`, but received `weights.shape={weights.shape}`"
                    f" and `x.shape={x.shape}`."
                )

        if weights is None or shape_match_on_axis or shape_match:
            return KerasTensor(
                reduce_shape(x.shape, axis=[self.axis]),
                dtype=x.dtype,
            )
        else:
            # `weights` can either be a 1D array of length `x.shape[axis]` or
            # of the same shape as `x`.
            raise ValueError(
                "`weights` must have the same size as `x` at "
                f"`axis={self.axis}` but received "
                f"`weights.shape={weights.shape}` while x.shape at "
                f"`{self.axis}` is `{x.shape[self.axis]}`."
            )


@keras_core_export(["keras_core.ops.average", "keras_core.ops.numpy.average"])
def average(x, axis=None, weights=None):
    """Compute the weighted average along the specified axis.

    Args:
        x: Input tensor.
        axis: Integer along which to average `x`. The default, `axis=None`,
            will average over all of the elements of the input tensor. If axis
            is negative it counts from the last to the first axis.
        weights: Tensor of wieghts associated with the values in `x`. Each
            value in `x` contributes to the average according to its
            associated weight. The weights array can either be 1-D (in which
            case its length must be the size of a along the given axis) or of
            the same shape as `x`. If `weights=None` (default), then all data
            in `x` are assumed to have a weight equal to one.

            The 1-D calculation is: `avg = sum(a * weights) / sum(weights)`.
            The only constraint on weights is that `sum(weights)` must not be 0.

    Returns:
        Return the average along the specified axis.

    Examples:
    >>> data = keras_core.ops.arange(1, 5)
    >>> data
    array([1, 2, 3, 4], dtype=int32)
    >>> keras_core.ops.average(data)
    array(2.5, dtype=float32)
    >>> keras_core.ops.average(
    ...     keras_core.ops.arange(1, 11),
    ...     weights=keras_core.ops.arange(10, 0, -1)
    ... )
    array(4., dtype=float32)

    >>> data = keras_core.ops.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]], dtype=int32)
    >>> keras_core.ops.average(
    ...     data,
    ...     axis=1,
    ...     weights=keras_core.ops.array([1./4, 3./4])
    ... )
    array([0.75, 2.75, 4.75], dtype=float32)
    >>> keras_core.ops.average(
    ...     data,
    ...     weights=keras_core.ops.array([1./4, 3./4])
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Axis must be specified when shapes of a and weights differ.
    """
    if any_symbolic_tensors((x,)):
        return Average(axis=axis).symbolic_call(x, weights=weights)
    return backend.numpy.average(x, weights=weights, axis=axis)


class Bincount(Operation):
    def __init__(self, weights=None, minlength=0):
        super().__init__()
        self.weights = weights
        self.minlength = minlength

    def call(self, x):
        return backend.numpy.bincount(
            x, weights=self.weights, minlength=self.minlength
        )

    def compute_output_spec(self, x):
        out_shape = backend.numpy.amax(x) + 1
        return KerasTensor(out_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.bincount", "keras_core.ops.numpy.bincount"])
def bincount(x, weights=None, minlength=0):
    """Count the number of occurrences of each value in a tensor of integers.

    Args:
        x: Input tensor.
            It must be of dimension 1, and it must only contain non-negative
            integer(s).
        weights: Weight tensor.
            It must have the same length as `x`. The default value is `None`.
            If specified, `x` is weighted by it, i.e. if `n = x[i]`,
            `out[n] += weight[i]` instead of the default behavior `out[n] += 1`.
        minlength: An integer.
            The default value is 0. If specified, there will be at least
            this number of bins in the output tensor. If greater than
            `max(x) + 1`, each value of the output at an index higher than
            `max(x)` is set to 0.

    Returns:
        1D tensor where each element gives the number of occurrence(s) of its
        index value in x. Its length is the maximum between `max(x) + 1` and
        minlength.

    Examples:
    >>> x = keras_core.ops.array([1, 2, 2, 3], dtype="uint8")
    >>> keras_core.ops.bincount(x)
    array([0, 1, 2, 1], dtype=int32)
    >>> weights = x / 2
    >>> weights
    array([0.5, 1., 1., 1.5], dtype=float64)
    >>> keras_core.ops.bincount(x, weights=weights)
    array([0., 0.5, 2., 1.5], dtype=float64)
    >>> minlength = (keras_core.ops.max(x).numpy() + 1) + 2 # 6
    >>> keras_core.ops.bincount(x, minlength=minlength)
    array([0, 1, 2, 1, 0, 0], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Bincount(weights=weights, minlength=minlength).symbolic_call(x)
    return backend.numpy.bincount(x, weights=weights, minlength=minlength)


class BroadcastTo(Operation):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def call(self, x):
        return backend.numpy.broadcast_to(x, self.shape)

    def compute_output_spec(self, x):
        # Catch broadcasting errors for clear error messages.
        broadcast_shapes(x.shape, self.shape)
        return KerasTensor(self.shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.broadcast_to",
        "keras_core.ops.numpy.broadcast_to",
    ]
)
def broadcast_to(x, shape):
    """Broadcast a tensor to a new shape.

    Args:
        x: The tensor to broadcast.
        shape: The shape of the desired tensor. A single integer `i` is
            interpreted as `(i,)`.

    Returns:
        A tensor with the desired shape.

    Examples:
    >>> x = keras_core.ops.array([1, 2, 3])
    >>> keras_core.ops.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    if any_symbolic_tensors((x,)):
        return BroadcastTo(shape=shape).symbolic_call(x)
    return backend.numpy.broadcast_to(x, shape)


class Ceil(Operation):
    def call(self, x):
        return backend.numpy.ceil(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.ceil", "keras_core.ops.numpy.ceil"])
def ceil(x):
    """Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.

    Args:
        x: Input tensor.

    Returns:
        The ceiling of each element in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Ceil().symbolic_call(x)
    return backend.numpy.ceil(x)


class Clip(Operation):
    def __init__(self, x_min, x_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def call(self, x):
        return backend.numpy.clip(x, self.x_min, self.x_max)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.clip", "keras_core.ops.numpy.clip"])
def clip(x, x_min, x_max):
    """Clip (limit) the values in a tensor.

    Given an interval, values outside the interval are clipped to the
    interval edges. For example, if an interval of `[0, 1]` is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.

    Args:
        x: Input tensor.
        x_min: Minimum value.
        x_max: Maximum value.
    Returns:
        The clipped tensor.
    """
    if any_symbolic_tensors((x,)):
        return Clip(x_min, x_max).symbolic_call(x)
    return backend.numpy.clip(x, x_min, x_max)


class Concatenate(Operation):
    def __init__(self, axis=0):
        super().__init__()
        if axis is None:
            raise ValueError("`axis` cannot be None for `concatenate`.")
        self.axis = axis

    def call(self, xs):
        return backend.numpy.concatenate(xs, axis=self.axis)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        for x in xs:
            if not shape_equal(
                x.shape, first_shape, axis=[self.axis], allow_none=True
            ):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[self.axis] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[self.axis]
        output_shape = list(first_shape)
        output_shape[self.axis] = total_size_on_axis
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.concatenate",
        "keras_core.ops.numpy.concatenate",
    ]
)
def concatenate(xs, axis=0):
    """Join a sequence of tensors along an existing axis.

    Args:
        xs: The sequence of tensors to concatenate.
        axis: The axis along which the tensors will be joined. Defaults to 0.

    Returns:
        The concatenated tensor.
    """
    if any_symbolic_tensors(xs):
        return Concatenate(axis=axis).symbolic_call(xs)
    return backend.numpy.concatenate(xs, axis=axis)


class Conjugate(Operation):
    def call(self, x):
        return backend.numpy.conjugate(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(
    ["keras_core.ops.conjugate", "keras_core.ops.numpy.conjugate"]
)
def conjugate(x):
    """Returns the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the sign
    of its imaginary part.

    `keras_core.ops.conj` is a shorthand for this function.

    Args:
        x: Input tensor.

    Returns:
        The complex conjugate of each element in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Conjugate().symbolic_call(x)
    return backend.numpy.conjugate(x)


class Conj(Conjugate):
    pass


@keras_core_export(["keras_core.ops.conj", "keras_core.ops.numpy.conj"])
def conj(x):
    """Shorthand for `keras_core.ops.conjugate`."""
    return conjugate(x)


class Copy(Operation):
    def call(self, x):
        return backend.numpy.copy(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.copy", "keras_core.ops.numpy.copy"])
def copy(x):
    """Returns a copy of `x`.

    Args:
        x: Input tensor.

    Returns:
        A copy of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Copy().symbolic_call(x)
    return backend.numpy.copy(x)


class Cos(Operation):
    def call(self, x):
        return backend.numpy.cos(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.cos", "keras_core.ops.numpy.cos"])
def cos(x):
    """Cosine, element-wise.

    Args:
        x: Input tensor.

    Returns:
        The corresponding cosine values.
    """
    if any_symbolic_tensors((x,)):
        return Cos().symbolic_call(x)
    return backend.numpy.cos(x)


class Cosh(Operation):
    def call(self, x):
        return backend.numpy.cosh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.cosh", "keras_core.ops.numpy.cosh"])
def cosh(x):
    """Hyperbolic cosine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Cosh().symbolic_call(x)
    return backend.numpy.cosh(x)


class CountNonzero(Operation):
    def __init__(self, axis=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = axis

    def call(self, x):
        return backend.numpy.count_nonzero(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis),
            dtype="int32",
        )


@keras_core_export(
    [
        "keras_core.ops.count_nonzero",
        "keras_core.ops.numpy.count_nonzero",
    ]
)
def count_nonzero(x, axis=None):
    """Counts the number of non-zero values in `x` along the given `axis`.

    If no axis is specified then all non-zeros in the tensor are counted.

    Args:
        x: Input tensor.
        axis: Axis or tuple of axes along which to count the number of
            non-zeros. Defaults to `None`.

    Returns:
        int or tensor of ints.

    Examples:
    >>> x = keras_core.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]])
    >>> keras_core.ops.count_nonzero(x)
    5
    >>> keras_core.ops.count_nonzero(x, axis=0)
    array([1, 1, 2, 1], dtype=int64)
    >>> keras_core.ops.count_nonzero(x, axis=1)
    array([2, 3], dtype=int64)
    """
    if any_symbolic_tensors((x,)):
        return CountNonzero(axis=axis).symbolic_call(x)
    return backend.numpy.count_nonzero(x, axis=axis)


class Cross(Operation):
    def __init__(self, axisa=-1, axisb=-1, axisc=-1, axis=None):
        super().__init__()
        if axis is not None:
            self.axisa = axis
            self.axisb = axis
            self.axisc = axis
        else:
            self.axisa = axisa
            self.axisb = axisb
            self.axisc = axisc

    def call(self, x1, x2):
        return backend.numpy.cross(x1, x2, self.axisa, self.axisb, self.axisc)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)

        x1_value_size = x1_shape[self.axisa]
        x2_value_size = x2_shape[self.axisa]
        del x1_shape[self.axisa]
        del x2_shape[self.axisb]
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        if x1_value_size is not None and x1_value_size not in (2, 3):
            raise ValueError(
                "`x1`'s dim on `axis={axisa}` must be either 2 or 3, but "
                f"received: {x1_value_size}"
            )
        if x2_value_size is not None and x2_value_size not in (2, 3):
            raise ValueError(
                "`x2`'s dim on `axis={axisb}` must be either 2 or 3, but "
                f"received: {x2_value_size}"
            )

        if x1_value_size == 3 or x2_value_size == 3:
            value_size = [3]
        else:
            value_size = []

        output_shape = (
            output_shape[: self.axisc] + value_size + output_shape[self.axisc :]
        )
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.cross", "keras_core.ops.numpy.cross"])
def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Returns the cross product of two (arrays of) vectors.

    The cross product of `x1` and `x2` in R^3 is a vector
    perpendicular to both `x1` and `x2`. If `x1` and `x2` are arrays of
    vectors, the vectors are defined by the last axis of `x1` and `x2`
    by default, and these axes can have dimensions 2 or 3.

    Where the dimension of either `x1` or `x2` is 2, the third component of
    the input vector is assumed to be zero and the cross product calculated
    accordingly.

    In cases where both input vectors have dimension 2, the z-component of
    the cross product is returned.

    Args:
        x1: Components of the first vector(s).
        x2: Components of the second vector(s).
        axisa: Axis of `x1` that defines the vector(s). Defaults to -1.
        axisb: Axis of `x2` that defines the vector(s). Defaults to -1.
        axisc: Axis of the result containing the cross product vector(s).
            Ignored if both input vectors have dimension 2, as the return is
            scalar. By default, the last axis.
        axis: If defined, the axis of `x1`, `x2` and the result that
            defines the vector(s) and cross product(s). Overrides `axisa`,
            `axisb` and `axisc`.

    Note:
        Torch backend does not support two dimensional vectors, or the
        arguments `axisa`, `axisb` and `axisc`. Use `axis` instead.

    Returns:
        Vector cross product(s).
    """
    if any_symbolic_tensors((x1, x2)):
        return Cross(
            axisa=axisa, axisb=axisb, axisc=axisc, axis=axis
        ).symbolic_call(x1, x2)
    return backend.numpy.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


class Cumprod(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.cumprod(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
            return KerasTensor(output_shape, dtype="int32")
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.cumprod", "keras_core.ops.numpy.cumprod"])
def cumprod(x, axis=None):
    """Return the cumulative product of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative product is computed.
            By default the input is flattened.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Cumprod(axis=axis).symbolic_call(x)
    return backend.numpy.cumprod(x, axis=axis)


class Cumsum(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.cumsum(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
            return KerasTensor(output_shape, dtype="int32")
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.cumsum", "keras_core.ops.numpy.cumsum"])
def cumsum(x, axis=None):
    """Returns the cumulative sum of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative sum is computed.
            By default the input is flattened.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Cumsum(axis=axis).symbolic_call(x)
    return backend.numpy.cumsum(x, axis=axis)


class Diag(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.diag(x, k=self.k)

    def compute_output_spec(self, x):
        x_shape = x.shape
        if len(x_shape) == 1:
            if x_shape[0] is None:
                output_shape = [None, None]
            else:
                output_shape = [
                    x_shape[0] + int(np.abs(self.k)),
                    x_shape[0] + int(np.abs(self.k)),
                ]
        elif len(x_shape) == 2:
            if None in x_shape:
                output_shape = [None]
            else:
                shorter_side = np.minimum(x_shape[0], x_shape[1])
                if self.k > 0:
                    remaining = x_shape[1] - self.k
                else:
                    remaining = x_shape[0] + self.k
                output_shape = [
                    int(np.maximum(0, np.minimum(remaining, shorter_side)))
                ]
        else:
            raise ValueError(
                f"`x` must be 1-D or 2-D, but received shape {x.shape}."
            )
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.diag", "keras_core.ops.numpy.diag"])
def diag(x, k=0):
    """Extract a diagonal or construct a diagonal array.

    Args:
        x: Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`.
            If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
        k: The diagonal to consider. Defaults to 0. Use `k > 0` for diagonals
            above the main diagonal, and `k < 0` for diagonals below
            the main diagonal.

    Returns:
        The extracted diagonal or constructed diagonal tensor.

    Examples:
    >>> from keras_core import ops
    >>> x = ops.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> ops.diag(x)
    array([0, 4, 8])
    >>> ops.diag(x, k=1)
    array([1, 5])
    >>> ops.diag(x, k=-1)
    array([3, 7])

    >>> ops.diag(ops.diag(x)))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """
    if any_symbolic_tensors((x,)):
        return Diag(k=k).symbolic_call(x)
    return backend.numpy.diag(x, k=k)


class Diagonal(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1):
        super().__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.diagonal(
            x,
            offset=self.offset,
            axis1=self.axis1,
            axis2=self.axis2,
        )

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if len(x_shape) < 2:
            raise ValueError(
                "`diagonal` requires an array of at least two dimensions, but "
                "`x` is of shape {x.shape}."
            )

        shape_2d = [x_shape[self.axis1], x_shape[self.axis2]]
        x_shape[self.axis1] = -1
        x_shape[self.axis2] = -1
        output_shape = list(filter((-1).__ne__, x_shape))
        if None in shape_2d:
            diag_shape = [None]
        else:
            shorter_side = np.minimum(shape_2d[0], shape_2d[1])
            if self.offset > 0:
                remaining = shape_2d[1] - self.offset
            else:
                remaining = shape_2d[0] + self.offset
            diag_shape = [
                int(np.maximum(0, np.minimum(remaining, shorter_side)))
            ]
        output_shape = output_shape + diag_shape
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.diagonal", "keras_core.ops.numpy.diagonal"])
def diagonal(x, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the
    collection of elements of the form `x[i, i+offset]`.

    If `x` has more than two dimensions, the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal
    is returned.

    The shape of the resulting array can be determined by removing `axis1`
    and `axis2` and appending an index to the right equal to the size of
    the resulting diagonals.

    Args:
        x: Input tensor.
        offset: Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to 0 (main diagonal).
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
            Defaults to 0 (first axis).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.
            Defaults to 1 (second axis).

    Returns:
        Tensor of diagonals.

    Examples:
    >>> from keras_core import ops
    >>> x = ops.arange(4).reshape((2, 2))
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.diagonal()
    array([0, 3])
    >>> x.diagonal(1)
    array([1])

    >>> x = ops.arange(8).reshape((2, 2, 2))
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> x.diagonal(0, 0, 1)
    array([[0, 6],
           [1, 7]])
    """
    if any_symbolic_tensors((x,)):
        return Diagonal(
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        ).symbolic_call(x)
    return backend.numpy.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


class Digitize(Operation):
    def call(self, x, bins):
        return backend.numpy.digitize(x, bins)

    def compute_output_spec(self, x, bins):
        bins_shape = bins.shape
        if len(bins_shape) > 1:
            raise ValueError(
                f"`bins` must be a 1D array. Received: bins={bins} "
                f"with shape bins.shape={bins_shape}"
            )
        return KerasTensor(x.shape, dtype="int32")


@keras_core_export(["keras_core.ops.digitize", "keras_core.ops.numpy.digitize"])
def digitize(x, bins):
    """Returns the indices of the bins to which each value in `x` belongs.

    Args:
        x: Input array to be binned.
        bins: Array of bins. It has to be one-dimensional and monotonically
            increasing.

    Returns:
        Output array of indices, of same shape as `x`.

    Example:
    >>> x = np.array([0.0, 1.0, 3.0, 1.6])
    >>> bins = np.array([0.0, 3.0, 4.5, 7.0])
    >>> keras_core.ops.digitize(x, bins)
    array([1, 1, 2, 1])
    """
    if any_symbolic_tensors((x, bins)):
        return Digitize().symbolic_call(x, bins)
    return backend.numpy.digitize(x, bins)


class Dot(Operation):
    def call(self, x1, x2):
        return backend.numpy.dot(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        if x1_shape == [] or x2_shape == []:
            return multiply(x1, x2)
        if len(x1_shape) == 1 and len(x2_shape) == 1:
            return KerasTensor([], dtype=x1.dtype)
        if len(x2_shape) == 1:
            if x1_shape[-1] != x2_shape[0]:
                raise ValueError(
                    "Shape must match on the last axis of `x1` and `x2` when "
                    "`x1` is N-d array while `x2` is 1-D, but receive shape "
                    f"`x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
                )
            return KerasTensor(x1_shape[:-1], dtype=x1.dtype)

        if (
            x1_shape[-1] is None
            or x2_shape[-2] is None
            or x1_shape[-1] == x2_shape[-2]
        ):
            del x1_shape[-1]
            del x2_shape[-2]
            return KerasTensor(x1_shape + x2_shape, dtype=x1.dtype)

        raise ValueError(
            "Shape must match on the last axis of `x1` and second last "
            "axis of `x2` when `x1` is N-d array while `x2` is M-D, but "
            f"received `x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
        )


@keras_core_export(["keras_core.ops.dot", "keras_core.ops.numpy.dot"])
def dot(x1, x2):
    """Dot product of two tensors.

    - If both `x1` and `x2` are 1-D tensors, it is inner product of vectors
      (without complex conjugation).
    - If both `x1` and `x2` are 2-D tensors, it is matrix multiplication.
    - If either `x1` or `x2` is 0-D (scalar), it is equivalent to `x1 * x2`.
    - If `x1` is an N-D tensor and `x2` is a 1-D tensor, it is a sum product
      over the last axis of `x1` and `x2`.
    - If `x1` is an N-D tensor and `x2` is an M-D tensor (where `M>=2`),
      it is a sum product over the last axis of `x1` and the second-to-last
      axis of `x2`: `dot(x1, x2)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`.

    Args:
        x1: First argument.
        x2: Second argument.

    Note:
        Torch backend does not accept 0-D tensors as arguments.

    Returns:
        Dot product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Dot().symbolic_call(x1, x2)
    return backend.numpy.dot(x1, x2)


class Einsum(Operation):
    def __init__(self, subscripts):
        super().__init__()
        self.subscripts = subscripts

    def call(self, *operands):
        return backend.numpy.einsum(self.subscripts, *operands)

    def compute_output_spec(self, *operands):
        """Compute the output shape of `einsum`.

        The shape computation follows the steps below:
        1. Find all letters in the input specs (left part of "->"), and
            break them into two categories: letters appearing more than once
            go to `reduced_dims`, otherwise go to `kept_dims`.
        2. Adjust `reduced_dims` and `kept_dims` based on the output spec
            (right part of "->"). The rule is if the letter appears in the
            output spec, then move it to `kept_dims`, otherwise move it to
            `reduced_dims`.
        3. Compute the target output shape. If no output spec is set, then
            the target output shape will be "...{kept_dims}", e.g., "...ijk",
            else it will be the same as output spec. "..." is a wildcard that
            could map shape of arbitrary length.
        4. For each operand in `operands`, map the shape specified in the input
            spec to the output target, e.g, if operand is of shape [2,3,4],
            input spec is "i..." and output target is "i...jk", then 2 will go
            the index 0. For dims not represented by any letter, insert to the
            wildcard part. For each letter in output target not appearing in
            input spec, the dim will be 1 for broadcasting. After 4, each
            operand should have a target shape containing only number and
            `None`.
        5. Broadcast all shapes computed from 4, and the result is the output
            shape.

        Let's take an example to illustrate the steps above. Let's define:
        ```python
        x = KerasTensor([None, 3, 4])
        y = KerasTensor(2, 4, 3)
        z = knp.einsum("...ij, kji->...k", x, y)
        ```

        1. `reduced_dims` is {"i", "j"}, `kept_dims` is {"k"}.
        2. `reduced_dims` is still {"i", "j"}, and `kept_dims` is {"k"}.
        3. Output target is "...k".
        4. For `x`, the input spec is "...ij", and the output target is "...k".
            "i" and "j" do not appear in the output target, so no replacement
            happens, and [None] goes to wildcard. Afterwards, "k" is replaced
            by 1, so we get shape [None, 1]. Applying the same logic to `y`, we
            get shape [2].
        5. Broadcast [None, 1] and [2], and we get [None, 2], which is the
            output shape.
        """
        split_subscripts = self.subscripts.split("->")
        if len(split_subscripts) > 2:
            raise ValueError(
                "At most one '->' is supported in `einsum` subscripts, but "
                f"received {self.subscripts}."
            )
        if len(split_subscripts) == 2:
            subscripts = split_subscripts[0]
            output_spec = split_subscripts[1]
        else:
            subscripts = self.subscripts
            output_spec = None
        input_specs = subscripts.split(",")
        if len(input_specs) != len(operands):
            raise ValueError(
                f"Number of operands ({len(operands)}) does not match the "
                f"number of input specs ({len(input_specs)}) in `einsum`, "
                f"received subscripts={self.subscripts}."
            )
        reduced_dims = set()
        kept_dims = set()
        for s in subscripts:
            if not s.isalpha():
                continue
            if s not in reduced_dims and s not in kept_dims:
                kept_dims.add(s)
            elif s in kept_dims:
                kept_dims.remove(s)
                reduced_dims.add(s)

        if output_spec is not None:
            # The output spec changes the rule of kept_dims and reduced_dims.
            # In short, dims appearing in the output spec will be kept, and
            # dims not appearing in the output spec will be reduced.
            kept_dims_copy = kept_dims.copy()
            reduced_dims_copy = reduced_dims.copy()
            for dim in kept_dims:
                if dim not in output_spec:
                    kept_dims_copy.remove(dim)
                    reduced_dims_copy.add(dim)
            for dim in reduced_dims:
                if dim in output_spec:
                    reduced_dims_copy.remove(dim)
                    kept_dims_copy.add(dim)
            kept_dims = kept_dims_copy
            reduced_dims = reduced_dims_copy

        reduced_dims = sorted(reduced_dims)
        kept_dims = sorted(kept_dims)

        if output_spec is None:
            target_broadcast_spec = "..." + "".join(kept_dims)
        else:
            target_broadcast_spec = output_spec

        expanded_operands_shapes = []
        for x, spec in zip(operands, input_specs):
            x_shape = getattr(x, "shape", [])
            x_shape = [-1 if size is None else size for size in x_shape]
            split_spec = spec.split("...")
            expanded_shape = target_broadcast_spec
            if len(split_spec) == 1:
                # In this case, the input spec is just a string of letters,
                # e.g., "ijk".
                if len(x_shape) != len(split_spec[0]):
                    raise ValueError(
                        "Number of dimensions in the subscript does not "
                        "match the number of dimensions in the operand, "
                        f"received subscript `{spec}` and operand of shape "
                        f"{x_shape}."
                    )
                for size, s in zip(x_shape, split_spec[0]):
                    # Replace the letter with the right shape.
                    expanded_shape = expanded_shape.replace(s, str(size) + " ")
                expanded_shape = expanded_shape.replace("...", "")
            else:
                # In this case, the input spec has "...", e.g., "i...j", "i...",
                # or "...j".
                for i in range(len(split_spec[0])):
                    expanded_shape = expanded_shape.replace(
                        split_spec[0][i], str(x_shape[i]) + " "
                    )
                for i in range(len(split_spec[1])):
                    expanded_shape = expanded_shape.replace(
                        split_spec[1][-i - 1], str(x_shape[-i - 1]) + " "
                    )
                # Shape matched by "..." will be inserted to the position of
                # "...".
                wildcard_shape_start_index = len(split_spec[0])
                wildcard_shape_end_index = (
                    len(x_shape)
                    if len(split_spec[1]) == 0
                    else -len(split_spec[1])
                )
                wildcard_shape = x_shape[
                    wildcard_shape_start_index:wildcard_shape_end_index
                ]
                wildcard_shape_str = (
                    " ".join([str(size) for size in wildcard_shape]) + " "
                )
                expanded_shape = expanded_shape.replace(
                    "...", wildcard_shape_str
                )
            # Replace all letters not yet handled with "1" for broadcasting.
            expanded_shape = re.sub("[a-z]", "1 ", expanded_shape)
            expanded_shape = expanded_shape.split()
            expanded_shape = [
                None if size == "-1" else int(size) for size in expanded_shape
            ]
            expanded_operands_shapes.append(expanded_shape)

        output_shape = expanded_operands_shapes[0]
        for shape in expanded_operands_shapes[1:]:
            output_shape = broadcast_shapes(output_shape, shape)
        dtype = None
        for x in operands:
            if hasattr(x, "dtype"):
                dtype = x.dtype
                break
        return KerasTensor(output_shape, dtype=dtype)


@keras_core_export(["keras_core.ops.einsum", "keras_core.ops.numpy.einsum"])
def einsum(subscripts, *operands):
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts: Specifies the subscripts for summation as comma separated
            list of subscript labels. An implicit (classical Einstein
            summation) calculation is performed unless the explicit indicator
            `->` is included as well as subscript labels of the precise
            output form.
        operands: The operands to compute the Einstein sum of.

    Returns:
        The calculation based on the Einstein summation convention.

    Example:
    >>> from keras_core import ops
    >>> a = ops.arange(25).reshape(5, 5)
    >>> b = ops.arange(5)
    >>> c = ops.arange(6).reshape(2, 3)

    Trace of a matrix:

    >>> ops.einsum("ii", a)
    60
    >>> ops.einsum(a, [0, 0])
    60
    >>> ops.trace(a)
    60

    Extract the diagonal:

    >>> ops.einsum("ii -> i", a)
    array([ 0,  6, 12, 18, 24])
    >>> ops.einsum(a, [0, 0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> ops.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis:

    >>> ops.einsum("ij -> i", a)
    array([ 10,  35,  60,  85, 110])
    >>> ops.einsum(a, [0, 1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> ops.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional tensors summing a single axis can be done
    with ellipsis:

    >>> ops.einsum("...j -> ...", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [..., 1], [...])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose or reorder any number of axes:

    >>> ops.einsum("ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.einsum("ij -> ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.einsum(c, [1, 0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Matrix vector multiplication:

    >>> ops.einsum("ij, j", a, b)
    array([ 30,  80, 130, 180, 230])
    >>> ops.einsum(a, [0, 1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> ops.einsum("...j, j", a, b)
    array([ 30,  80, 130, 180, 230])
    """
    if any_symbolic_tensors(operands):
        return Einsum(subscripts).symbolic_call(*operands)
    return backend.numpy.einsum(subscripts, *operands)


class Empty(Operation):
    def call(self, shape, dtype="float32"):
        return backend.numpy.empty(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


@keras_core_export(["keras_core.ops.empty", "keras_core.ops.numpy.empty"])
def empty(shape, dtype="float32"):
    """Return a tensor of given shape and type filled with uninitialized data.

    Args:
        shape: Shape of the empty tensor.
        dtype: Desired data type of the empty tensor.

    Returns:
        The empty tensor.
    """
    return backend.numpy.empty(shape, dtype=dtype)


class Equal(Operation):
    def call(self, x1, x2):
        return backend.numpy.equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.equal", "keras_core.ops.numpy.equal"])
def equal(x1, x2):
    """Returns `(x1 == x2)` element-wise.

    Args:
        x1: Tensor to compare.
        x2: Tensor to compare.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Equal().symbolic_call(x1, x2)
    return backend.numpy.equal(x1, x2)


class Exp(Operation):
    def call(self, x):
        return backend.numpy.exp(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.exp", "keras_core.ops.numpy.exp"])
def exp(x):
    """Calculate the exponential of all elements in the input tensor.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise exponential of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Exp().symbolic_call(x)
    return backend.numpy.exp(x)


class ExpandDims(Operation):
    def __init__(self, axis):
        super().__init__()
        if isinstance(axis, list):
            raise ValueError(
                "The `axis` argument to `expand_dims` should be an integer, "
                f"but received a list: {axis}."
            )
        self.axis = axis

    def call(self, x):
        return backend.numpy.expand_dims(x, self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if self.axis < 0:
            axis = len(x.shape) + 1 + self.axis
        else:
            axis = self.axis
        output_shape = x_shape[:axis] + [1] + x_shape[axis:]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.expand_dims",
        "keras_core.ops.numpy.expand_dims",
    ]
)
def expand_dims(x, axis):
    """Expand the shape of a tensor.

    Insert a new axis at the `axis` position in the expanded tensor shape.

    Args:
        x: Input tensor.
        axis: Position in the expanded axes where the new axis
            (or axes) is placed.

    Returns:
        Output tensor with the number of dimensions increased.
    """
    if any_symbolic_tensors((x,)):
        return ExpandDims(axis=axis).symbolic_call(x)
    return backend.numpy.expand_dims(x, axis)


class Expm1(Operation):
    def call(self, x):
        return backend.numpy.expm1(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.expm1", "keras_core.ops.numpy.expm1"])
def expm1(x):
    """Calculate `exp(x) - 1` for all elements in the tensor.

    Args:
        x: Input values.

    Returns:
        Output tensor, element-wise exponential minus one.
    """
    if any_symbolic_tensors((x,)):
        return Expm1().symbolic_call(x)
    return backend.numpy.expm1(x)


class Flip(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.flip(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.flip", "keras_core.ops.numpy.flip"])
def flip(x, axis=None):
    """Reverse the order of elements in the tensor along the given axis.

    The shape of the tensor is preserved, but the elements are reordered.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to flip the tensor. The default,
            `axis=None`, will flip over all of the axes of the input tensor.

    Returns:
        Output tensor with entries of `axis` reversed.
    """
    if any_symbolic_tensors((x,)):
        return Flip(axis=axis).symbolic_call(x)
    return backend.numpy.flip(x, axis=axis)


class Floor(Operation):
    def call(self, x):
        return backend.numpy.floor(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.floor", "keras_core.ops.numpy.floor"])
def floor(x):
    """Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that `i <= x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise floor of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Floor().symbolic_call(x)
    return backend.numpy.floor(x)


class Full(Operation):
    def call(self, shape, fill_value, dtype=None):
        return backend.numpy.full(shape, fill_value, dtype=dtype)

    def compute_output_spec(self, shape, fill_value, dtype=None):
        return KerasTensor(shape, dtype=dtype)


@keras_core_export(["keras_core.ops.full", "keras_core.ops.numpy.full"])
def full(shape, fill_value, dtype=None):
    """Return a new tensor of given shape and type, filled with `fill_value`.

    Args:
        shape: Shape of the new tensor.
        fill_value: Fill value.
        dtype: Desired data type of the tensor.

    Returns:
        Output tensor.
    """
    return backend.numpy.full(shape, fill_value, dtype=dtype)


class FullLike(Operation):
    def call(self, x, fill_value, dtype=None):
        return backend.numpy.full_like(x, fill_value, dtype=dtype)

    def compute_output_spec(self, x, fill_value, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


@keras_core_export(
    ["keras_core.ops.full_like", "keras_core.ops.numpy.full_like"]
)
def full_like(x, fill_value, dtype=None):
    """Return a full tensor with the same shape and type as the given tensor.

    Args:
        x: Input tensor.
        fill_value: Fill value.
        dtype: Overrides data type of the result.

    Returns:
        Tensor of `fill_value` with the same shape and type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return FullLike().symbolic_call(x, fill_value, dtype=dtype)
    return backend.numpy.full_like(x, fill_value, dtype=dtype)


class GetItem(Operation):
    def call(self, x, key):
        return x[key]

    def compute_output_spec(self, x, key):
        remaining_shape = list(x.shape)
        new_shape = []
        if isinstance(key, int):
            remaining_key = [key]
        elif isinstance(key, tuple):
            remaining_key = list(key)
        else:
            raise ValueError(
                f"Unsupported key type for array slice. Recieved: `{key}`"
            )
        num_ellipses = remaining_key.count(Ellipsis)
        if num_ellipses > 1:
            raise ValueError(
                f"Slice should only have one ellipsis. Recieved: `{key}`"
            )
        elif num_ellipses == 0:
            # Add an implicit final ellipsis.
            remaining_key.append(Ellipsis)
        # Consume slice key element by element.
        while True:
            if not remaining_key:
                break
            subkey = remaining_key.pop(0)
            # Check for `newaxis` and `Ellipsis`.
            if subkey == Ellipsis:
                # Keep as many slices remain in our key, omitting `newaxis`.
                needed = len(remaining_key) - remaining_key.count(np.newaxis)
                consumed = len(remaining_shape) - needed
                new_shape += remaining_shape[:consumed]
                remaining_shape = remaining_shape[consumed:]
                continue
            # All frameworks follow numpy for newaxis. `np.newaxis == None`.
            if subkey == np.newaxis:
                new_shape.append(1)
                continue
            # At this point, we need to consume a new axis from the shape.
            if not remaining_shape:
                raise ValueError(
                    f"Array has shape {x.shape} but slice "
                    f"has to many indices. Recieved: `{key}`"
                )
            length = remaining_shape.pop(0)
            if isinstance(subkey, int):
                if length is not None:
                    index = subkey if subkey >= 0 else subkey + length
                    if index < 0 or index >= length:
                        raise ValueError(
                            f"Array has shape {x.shape} but out-of-bounds "
                            f"index {key} was requested."
                        )
            elif isinstance(subkey, slice):
                if length is not None:
                    # python3 friendly way to compute a slice length.
                    new_length = len(range(*subkey.indices(length)))
                    new_shape.append(new_length)
                else:
                    new_shape.append(length)
            else:
                raise ValueError(
                    f"Unsupported key type for array slice. Recieved: `{key}`"
                )
        return KerasTensor(tuple(new_shape), dtype=x.dtype)


@keras_core_export(["keras_core.ops.get_item", "keras_core.ops.numpy.get_item"])
def get_item(x, key):
    """Return `x[key]`."""
    if any_symbolic_tensors((x,)):
        return GetItem().symbolic_call(x, key)
    return x[key]


class Greater(Operation):
    def call(self, x1, x2):
        return backend.numpy.greater(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.greater", "keras_core.ops.numpy.greater"])
def greater(x1, x2):
    """Return the truth value of `x1 > x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Greater().symbolic_call(x1, x2)
    return backend.numpy.greater(x1, x2)


class GreaterEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.greater_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    [
        "keras_core.ops.greater_equal",
        "keras_core.ops.numpy.greater_equal",
    ]
)
def greater_equal(x1, x2):
    """Return the truth value of `x1 >= x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return GreaterEqual().symbolic_call(x1, x2)
    return backend.numpy.greater_equal(x1, x2)


class Hstack(Operation):
    def call(self, xs):
        return backend.numpy.hstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[1], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[1] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[1]
        output_shape = list(first_shape)
        output_shape[1] = total_size_on_axis
        return KerasTensor(output_shape)


@keras_core_export(["keras_core.ops.hstack", "keras_core.ops.numpy.hstack"])
def hstack(xs):
    """Stack tensors in sequence horizontally (column wise).

    This is equivalent to concatenation along the first axis for 1-D tensors,
    and along the second axis for all other tensors.

    Args:
        xs: Sequence of tensors.

    Returns:
        The tensor formed by stacking the given tensors.
    """
    if any_symbolic_tensors((xs,)):
        return Hstack().symbolic_call(xs)
    return backend.numpy.hstack(xs)


class Identity(Operation):
    def call(self, n, dtype="float32"):
        return backend.numpy.identity(n, dtype=dtype)

    def compute_output_spec(self, n, dtype="float32"):
        return KerasTensor([n, n], dtype=dtype)


@keras_core_export(["keras_core.ops.identity", "keras_core.ops.numpy.identity"])
def identity(n, dtype="float32"):
    """Return the identity tensor.

    The identity tensor is a square tensor with ones on the main diagonal and
    zeros elsewhere.

    Args:
        n: Number of rows (and columns) in the `n x n` output tensor.
        dtype: Data type of the output tensor.

    Returns:
        The identity tensor.
    """
    return backend.numpy.identity(n, dtype=dtype)


class Imag(Operation):
    def call(self, x):
        return backend.numpy.imag(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.imag", "keras_core.ops.numpy.imag"])
def imag(x):
    """Return the imaginary part of the complex argument.

    Args:
        x: Input tensor.

    Returns:
        The imaginary component of the complex argument.
    """
    if any_symbolic_tensors((x,)):
        return Imag().symbolic_call(x)
    return backend.numpy.imag(x)


class Isclose(Operation):
    def call(self, x1, x2):
        return backend.numpy.isclose(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.isclose", "keras_core.ops.numpy.isclose"])
def isclose(x1, x2):
    """Return whether two tensors are element-wise almost equal.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return Isclose().symbolic_call(x1, x2)
    return backend.numpy.isclose(x1, x2)


class Isfinite(Operation):
    def call(self, x):
        return backend.numpy.isfinite(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_core_export(["keras_core.ops.isfinite", "keras_core.ops.numpy.isfinite"])
def isfinite(x):
    """Return whether a tensor is finite, element-wise.

    Real values are finite when they are not NaN, not positive infinity, and
    not negative infinity. Complex values are finite when both their real
    and imaginary parts are finite.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isfinite().symbolic_call(x)
    return backend.numpy.isfinite(x)


class Isinf(Operation):
    def call(self, x):
        return backend.numpy.isinf(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_core_export(["keras_core.ops.isinf", "keras_core.ops.numpy.isinf"])
def isinf(x):
    """Test element-wise for positive or negative infinity.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isinf().symbolic_call(x)
    return backend.numpy.isinf(x)


class Isnan(Operation):
    def call(self, x):
        return backend.numpy.isnan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_core_export(["keras_core.ops.isnan", "keras_core.ops.numpy.isnan"])
def isnan(x):
    """Test element-wise for NaN and return result as a boolean tensor.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isnan().symbolic_call(x)
    return backend.numpy.isnan(x)


class Less(Operation):
    def call(self, x1, x2):
        return backend.numpy.less(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.less", "keras_core.ops.numpy.less"])
def less(x1, x2):
    """Return the truth value of `x1 < x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Less().symbolic_call(x1, x2)
    return backend.numpy.less(x1, x2)


class LessEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.less_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    [
        "keras_core.ops.less_equal",
        "keras_core.ops.numpy.less_equal",
    ]
)
def less_equal(x1, x2):
    """Return the truth value of `x1 <= x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return LessEqual().symbolic_call(x1, x2)
    return backend.numpy.less_equal(x1, x2)


class Linspace(Operation):
    def __init__(
        self, num=50, endpoint=True, retstep=False, dtype=float, axis=0
    ):
        super().__init__()
        self.num = num
        self.endpoint = endpoint
        self.retstep = retstep
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.numpy.linspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            retstep=self.retstep,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        if self.axis == -1:
            output_shape = output_shape + [self.num]
        elif self.axis >= 0:
            output_shape = (
                output_shape[: self.axis]
                + [self.num]
                + output_shape[self.axis :]
            )
        else:
            output_shape = (
                output_shape[: self.axis + 1]
                + [self.num]
                + output_shape[self.axis + 1 :]
            )

        dtype = self.dtype if self.dtype is not None else start.dtype
        if self.retstep:
            return (KerasTensor(output_shape, dtype=dtype), None)
        return KerasTensor(output_shape, dtype=dtype)


@keras_core_export(["keras_core.ops.linspace", "keras_core.ops.numpy.linspace"])
def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    """Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the interval
    `[start, stop]`.

    The endpoint of the interval can optionally be excluded.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence, unless `endpoint` is set to
            `False`. In that case, the sequence consists of all but the last
            of `num + 1` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is `False`.
        num: Number of samples to generate. Default is 50. Must be
            non-negative.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is
            not included. Default is `True`.
        retstep: If `True`, return `(samples, step)`, where `step` is the
            spacing between samples.
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only if
            start or stop are array-like. Default is 0.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of evenly spaced numbers.
        If `retstep` is `True`, returns `(samples, step)`
    """
    if any_symbolic_tensors((start, stop)):
        return Linspace(num, endpoint, retstep, dtype, axis)(start, stop)
    return backend.numpy.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


class Log(Operation):
    def call(self, x):
        return backend.numpy.log(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.log", "keras_core.ops.numpy.log"])
def log(x):
    """Natural logarithm, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise natural logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log().symbolic_call(x)
    return backend.numpy.log(x)


class Log10(Operation):
    def call(self, x):
        return backend.numpy.log10(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.log10", "keras_core.ops.numpy.log10"])
def log10(x):
    """Return the base 10 logarithm of the input tensor, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base 10 logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log10().symbolic_call(x)
    return backend.numpy.log10(x)


class Log1p(Operation):
    def call(self, x):
        return backend.numpy.log1p(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.log1p", "keras_core.ops.numpy.log1p"])
def log1p(x):
    """Returns the natural logarithm of one plus the `x`, element-wise.

    Calculates `log(1 + x)`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise natural logarithm of `1 + x`.
    """
    if any_symbolic_tensors((x,)):
        return Log1p().symbolic_call(x)
    return backend.numpy.log1p(x)


class Log2(Operation):
    def call(self, x):
        return backend.numpy.log2(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.log2", "keras_core.ops.numpy.log2"])
def log2(x):
    """Base-2 logarithm of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base-2 logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log2().symbolic_call(x)
    return backend.numpy.log2(x)


class Logaddexp(Operation):
    def call(self, x1, x2):
        return backend.numpy.logaddexp(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    ["keras_core.ops.logaddexp", "keras_core.ops.numpy.logaddexp"]
)
def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs.

    Calculates `log(exp(x1) + exp(x2))`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logarithm of the sum of exponentiations
        of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Logaddexp().symbolic_call(x1, x2)
    return backend.numpy.logaddexp(x1, x2)


class LogicalAnd(Operation):
    def call(self, x1, x2):
        return backend.numpy.logical_and(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    [
        "keras_core.ops.logical_and",
        "keras_core.ops.numpy.logical_and",
    ]
)
def logical_and(x1, x2):
    """Computes the element-wise logical AND of the given input tensors.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logical AND of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalAnd().symbolic_call(x1, x2)
    return backend.numpy.logical_and(x1, x2)


class LogicalNot(Operation):
    def call(self, x):
        return backend.numpy.logical_not(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.logical_not",
        "keras_core.ops.numpy.logical_not",
    ]
)
def logical_not(x):
    """Computes the element-wise NOT of the given input tensor.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise logical NOT of the input.
    """
    if any_symbolic_tensors((x,)):
        return LogicalNot().symbolic_call(x)
    return backend.numpy.logical_not(x)


class LogicalOr(Operation):
    def call(self, x1, x2):
        return backend.numpy.logical_or(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    [
        "keras_core.ops.logical_or",
        "keras_core.ops.numpy.logical_or",
    ]
)
def logical_or(x1, x2):
    """Computes the element-wise logical OR of the given input tensors.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logical OR of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalOr().symbolic_call(x1, x2)
    return backend.numpy.logical_or(x1, x2)


class Logspace(Operation):
    def __init__(self, num=50, endpoint=True, base=10, dtype=float, axis=0):
        super().__init__()
        self.num = num
        self.endpoint = endpoint
        self.base = base
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.numpy.logspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            base=self.base,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        if self.axis == -1:
            output_shape = output_shape + [self.num]
        elif self.axis >= 0:
            output_shape = (
                output_shape[: self.axis]
                + [self.num]
                + output_shape[self.axis :]
            )
        else:
            output_shape = (
                output_shape[: self.axis + 1]
                + [self.num]
                + output_shape[self.axis + 1 :]
            )

        dtype = self.dtype if self.dtype is not None else start.dtype
        return KerasTensor(output_shape, dtype=dtype)


@keras_core_export(["keras_core.ops.logspace", "keras_core.ops.numpy.logspace"])
def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    """Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at `base ** start` and ends with
    `base ** stop` (see `endpoint` below).

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence, unless `endpoint` is `False`.
            In that case, `num + 1` values are spaced over the interval in
            log-space, of which all but the last (a sequence of length `num`)
            are returned.
        num: Number of samples to generate. Default is 50.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is not
            included. Default is `True`.
        base: The base of the log space. Default is 10
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only
            if start or stop are array-like.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of evenly spaced samples on a log scale.
    """
    if any_symbolic_tensors((start, stop)):
        return Logspace(num, endpoint, base, dtype, axis)(start, stop)
    return backend.numpy.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


class Matmul(Operation):
    def call(self, x1, x2):
        return backend.numpy.matmul(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        if len(x1_shape) == 1:
            x1_shape = (1, x1_shape[0])
        if len(x2_shape) == 1:
            x2_shape = (x2_shape[0], 1)
        if (
            x1_shape[-1] is not None
            and x2_shape[-2] is not None
            and x1_shape[-1] != x2_shape[-2]
        ):
            raise ValueError(
                "Inner dimensions (`x1.shape[-1]` and `x2.shape[-2]`) must be "
                f"equal, but received `x1.shape={x1.shape}` and "
                f"`x2.shape={x2.shape}`."
            )

        leading_shape = broadcast_shapes(x1_shape[:-2], x2_shape[:-2])
        last_2_dims_shape = [x1_shape[-2], x2_shape[-1]]
        output_shape = leading_shape + last_2_dims_shape
        if len(x1.shape) == 1:
            del output_shape[-2]
        if len(x2.shape) == 1:
            del output_shape[-1]
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.matmul", "keras_core.ops.numpy.matmul"])
def matmul(x1, x2):
    """Matrix product of two tensors.

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If either tensor is N-D, N > 2, it is treated as a stack of matrices
      residing in the last two indexes and broadcast accordingly.
    - If the first tensor is 1-D, it is promoted to a matrix by prepending
      a 1 to its dimensions. After matrix multiplication the prepended
      1 is removed.
    - If the second tensor is 1-D, it is promoted to a matrix by appending a 1
      to its dimensions. After matrix multiplication the appended 1 is removed.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, matrix product of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Matmul().symbolic_call(x1, x2)
    # The below conversion works around an outstanding JAX bug.
    x1 = backend.convert_to_tensor(x1)
    x2 = backend.convert_to_tensor(x2)
    return backend.numpy.matmul(x1, x2)


class Max(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.initial = initial

    def call(self, x):
        return backend.numpy.max(
            x, axis=self.axis, keepdims=self.keepdims, initial=self.initial
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.max", "keras_core.ops.numpy.max"])
def max(x, axis=None, keepdims=False, initial=None):
    """Return the maximum of a tensor or maximum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate. By default, flattened input
            is used.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Default is `False`.
        initial: The minimum value of an output element. Default is `None`.

    Returns:
        Maximum of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Max(axis=axis, keepdims=keepdims, initial=initial).symbolic_call(
            x
        )
    return backend.numpy.max(x, axis=axis, keepdims=keepdims, initial=initial)


class Maximum(Operation):
    def call(self, x1, x2):
        return backend.numpy.maximum(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.maximum", "keras_core.ops.numpy.maximum"])
def maximum(x1, x2):
    """Element-wise maximum of `x1` and `x2`.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise maximum of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Maximum().symbolic_call(x1, x2)
    return backend.numpy.maximum(x1, x2)


class Meshgrid(Operation):
    def __init__(self, indexing="xy"):
        super().__init__()
        if indexing not in ("xy", "ij"):
            raise ValueError(
                "Valid values for `indexing` are 'xy' and 'ij', "
                "but received {index}."
            )
        self.indexing = indexing

    def call(self, *x):
        return backend.numpy.meshgrid(*x, indexing=self.indexing)

    def compute_output_spec(self, *x):
        output_shape = []
        for xi in x:
            if len(xi.shape) == 0:
                size = 1
            else:
                if None in xi.shape:
                    size = None
                else:
                    size = int(np.prod(xi.shape))
            output_shape.append(size)
        if self.indexing == "ij":
            return [KerasTensor(output_shape) for _ in range(len(x))]
        tmp = output_shape[0]
        output_shape[0] = output_shape[1]
        output_shape[1] = tmp
        return [KerasTensor(output_shape) for _ in range(len(x))]


@keras_core_export(["keras_core.ops.meshgrid", "keras_core.ops.numpy.meshgrid"])
def meshgrid(*x, indexing="xy"):
    """Creates grids of coordinates from coordinate vectors.

    Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding
    lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors
    `G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output
    `Gi` is constructed by expanding `Ti` to the result shape.

    Args:
        x: 1-D tensors representing the coordinates of a grid.
        indexing: Cartesian (`"xy"`, default) or matrix (`"ij"`) indexing
            of output.

    Returns:
        Sequence of N tensors.

    Example:
    >>> from keras_core import ops
    >>> x = ops.array([1, 2, 3])
    >>> y = ops.array([4, 5, 6])

    >>> grid_x, grid_y = ops.meshgrid(x, y, indexing="ij")
    >>> grid_x
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> grid_y
    array([[4, 5, 6],
           [4, 5, 6],
           [4, 5, 6]])
    """
    if any_symbolic_tensors(x):
        return Meshgrid(indexing=indexing).symbolic_call(*x)
    return backend.numpy.meshgrid(*x, indexing=indexing)


class Min(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None):
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.initial = initial

    def call(self, x):
        return backend.numpy.min(
            x, axis=self.axis, keepdims=self.keepdims, initial=self.initial
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.min", "keras_core.ops.numpy.min"])
def min(x, axis=None, keepdims=False, initial=None):
    """Return the minimum of a tensor or minimum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate. By default, flattened input
            is used.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Default is `False`.
        initial: The maximum value of an output element. Default is `None`.

    Returns:
        Minimum of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Min(axis=axis, keepdims=keepdims, initial=initial).symbolic_call(
            x
        )
    return backend.numpy.min(x, axis=axis, keepdims=keepdims, initial=initial)


class Minimum(Operation):
    def call(self, x1, x2):
        return backend.numpy.minimum(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.minimum", "keras_core.ops.numpy.minimum"])
def minimum(x1, x2):
    """Element-wise minimum of `x1` and `x2`.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise minimum of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Minimum().symbolic_call(x1, x2)
    return backend.numpy.minimum(x1, x2)


class Mod(Operation):
    def call(self, x1, x2):
        return backend.numpy.mod(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.mod", "keras_core.ops.numpy.mod"])
def mod(x1, x2):
    """Returns the element-wise remainder of division.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise remainder of division.
    """
    if any_symbolic_tensors((x1, x2)):
        return Mod().symbolic_call(x1, x2)
    return backend.numpy.mod(x1, x2)


class Moveaxis(Operation):
    def __init__(self, source, destination):
        super().__init__()
        if isinstance(source, int):
            self.source = [source]
        else:
            self.source = source
        if isinstance(destination, int):
            self.destination = [destination]
        else:
            self.destination = destination

        if len(self.source) != len(self.destination):
            raise ValueError(
                "`source` and `destination` arguments must have the same "
                f"number of elements, but received `source={source}` and "
                f"`destination={destination}`."
            )

    def call(self, x):
        return backend.numpy.moveaxis(x, self.source, self.destination)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        output_shape = [-1 for _ in range(len(x.shape))]
        for sc, dst in zip(self.source, self.destination):
            output_shape[dst] = x_shape[sc]
            x_shape[sc] = -1
        i, j = 0, 0
        while i < len(output_shape):
            while i < len(output_shape) and output_shape[i] != -1:
                # Find the first dim unset.
                i += 1
            while j < len(output_shape) and x_shape[j] == -1:
                # Find the first dim not being passed.
                j += 1
            if i == len(output_shape):
                break
            output_shape[i] = x_shape[j]
            i += 1
            j += 1
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.moveaxis", "keras_core.ops.numpy.moveaxis"])
def moveaxis(x, source, destination):
    """Move axes of a tensor to new positions.

    Other axes remain in their original order.

    Args:
        x: Tensor whose axes should be reordered.
        source: Original positions of the axes to move. These must be unique.
        destination: Destinations positions for each of the original axes.
            These must also be unique.

    Returns:
        Tensor with moved axes.
    """
    if any_symbolic_tensors((x,)):
        return Moveaxis(source, destination).symbolic_call(x)
    return backend.numpy.moveaxis(x, source=source, destination=destination)


class NanToNum(Operation):
    def call(self, x):
        return backend.numpy.nan_to_num(x)


@keras_core_export(
    [
        "keras_core.ops.nan_to_num",
        "keras_core.ops.numpy.nan_to_num",
    ]
)
def nan_to_num(x):
    """Replace NaN with zero and infinity with large finite numbers.

    Args:
        x: Input data.

    Returns:
        `x`, with non-finite values replaced.
    """
    return backend.numpy.nan_to_num(x)


class Ndim(Operation):
    def call(self, x):
        return backend.numpy.ndim(
            x,
        )

    def compute_output_spec(self, x):
        return KerasTensor([len(x.shape)])


@keras_core_export(["keras_core.ops.ndim", "keras_core.ops.numpy.ndim"])
def ndim(x):
    """Return the number of dimensions of a tensor.

    Args:
        x: Input tensor.

    Returns:
        The number of dimensions in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Ndim().symbolic_call(x)
    return backend.numpy.ndim(x)


class Nonzero(Operation):
    def call(self, x):
        return backend.numpy.nonzero(x)


@keras_core_export(["keras_core.ops.nonzero", "keras_core.ops.numpy.nonzero"])
def nonzero(x):
    """Return the indices of the elements that are non-zero.

    Args:
        x: Input tensor.

    Returns:
        Indices of elements that are non-zero.
    """
    return backend.numpy.nonzero(x)


class NotEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.not_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    ["keras_core.ops.not_equal", "keras_core.ops.numpy.not_equal"]
)
def not_equal(x1, x2):
    """Return `(x1 != x2)` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparsion of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return NotEqual().symbolic_call(x1, x2)
    return backend.numpy.not_equal(x1, x2)


class OnesLike(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.ones_like(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_core_export(
    ["keras_core.ops.ones_like", "keras_core.ops.numpy.ones_like"]
)
def ones_like(x, dtype=None):
    """Return a tensor of ones with the same shape and type of `x`.

    Args:
        x: Input tensor.
        dtype: Overrides the data type of the result.

    Returns:
        A tensor of ones with the same shape and type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return OnesLike().symbolic_call(x, dtype=dtype)
    return backend.numpy.ones_like(x, dtype=dtype)


class ZerosLike(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.zeros_like(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_core_export(
    [
        "keras_core.ops.zeros_like",
        "keras_core.ops.numpy.zeros_like",
    ]
)
def zeros_like(x, dtype=None):
    """Return a tensor of zeros with the same shape and type as `x`.

    Args:
        x: Input tensor.
        dtype: Overrides the data type of the result.

    Returns:
        A tensor of zeros with the same shape and type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return ZerosLike().symbolic_call(x, dtype=dtype)
    return backend.numpy.zeros_like(x, dtype=dtype)


class Outer(Operation):
    def call(self, x1, x2):
        return backend.numpy.outer(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [1])
        x2_shape = getattr(x2, "shape", [1])
        if None in x1_shape:
            x1_flatten_shape = None
        else:
            x1_flatten_shape = int(np.prod(x1_shape))
        if None in x2_shape:
            x2_flatten_shape = None
        else:
            x2_flatten_shape = int(np.prod(x2_shape))
        output_shape = [x1_flatten_shape, x2_flatten_shape]
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.outer", "keras_core.ops.numpy.outer"])
def outer(x1, x2):
    """Compute the outer product of two vectors.

    Given two vectors `x1` and `x2`, the outer product is:

    ```
    out[i, j] = x1[i] * x2[j]
    ```

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Outer product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Outer().symbolic_call(x1, x2)
    return backend.numpy.outer(x1, x2)


class Pad(Operation):
    def __init__(self, pad_width, mode="constant"):
        super().__init__()
        self.pad_width = self._process_pad_width(pad_width)
        self.mode = mode

    def _process_pad_width(self, pad_width):
        if isinstance(pad_width, int):
            return ((pad_width, pad_width),)
        if isinstance(pad_width, (tuple, list)) and isinstance(
            pad_width[0], int
        ):
            return (pad_width,)
        first_len = len(pad_width[0])
        for i, pw in enumerate(pad_width):
            if len(pw) != first_len:
                raise ValueError(
                    "`pad_width` should be a list of tuples of length 2 or "
                    f"1, but received {pad_width}."
                )
            if len(pw) == 1:
                pad_width[i] = (pw[0], pw[0])
        return pad_width

    def call(self, x):
        return backend.numpy.pad(x, pad_width=self.pad_width, mode=self.mode)

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        if len(self.pad_width) == 1:
            pad_width = [self.pad_width[0] for _ in range(len(output_shape))]
        elif len(self.pad_width) == len(output_shape):
            pad_width = self.pad_width
        else:
            raise ValueError(
                "`pad_width` must have the same length as `x.shape`, but "
                f"received {len(self.pad_width)} and {len(x.shape)}."
            )

        for i in range(len(output_shape)):
            if output_shape[i] is None:
                output_shape[i] = None
            else:
                output_shape[i] += pad_width[i][0] + pad_width[i][1]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.pad", "keras_core.ops.numpy.pad"])
def pad(x, pad_width, mode="constant"):
    """Pad a tensor.

    Args:
        x: Tensor to pad.
        pad_width: Number of values padded to the edges of each axis.
            `((before_1, after_1), ...(before_N, after_N))` unique pad
            widths for each axis.
            `((before, after),)` yields same before and after pad for
            each axis.
            `(pad,)` or `int` is a shortcut for `before = after = pad`
            width for all axes.
        mode: One of `"constant"`, `"edge"`, `"linear_ramp"`,
            `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
            `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
            `"circular"`. Default is `"constant"`.

    Note:
        Torch backend only supports modes `"constant"`, `"reflect"`,
        `"symmetric"` and `"circular"`.
        Only Torch backend supports `"circular"` mode.

    Note:
        Tensorflow backend only supports modes `"constant"`, `"reflect"`
        and `"symmetric"`.

    Returns:
        Padded tensor.
    """
    if any_symbolic_tensors((x,)):
        return Pad(pad_width, mode=mode).symbolic_call(x)
    return backend.numpy.pad(x, pad_width, mode=mode)


class Prod(Operation):
    def __init__(self, axis=None, keepdims=False, dtype=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.dtype = dtype

    def call(self, x):
        return backend.numpy.prod(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
            dtype=self.dtype,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=self.dtype,
        )


@keras_core_export(["keras_core.ops.prod", "keras_core.ops.numpy.prod"])
def prod(x, axis=None, keepdims=False, dtype=None):
    """Return the product of tensor elements over a given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which a product is performed. The default,
            `axis=None`, will compute the product of all elements
            in the input tensor.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.
        dtype: Data type of the returned tensor.

    Returns:
        Product of elements of `x` over the given axis or axes.
    """
    if any_symbolic_tensors((x,)):
        return Prod(axis=axis, keepdims=keepdims, dtype=dtype).symbolic_call(x)
    return backend.numpy.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


class Ravel(Operation):
    def call(self, x):
        return backend.numpy.ravel(x)

    def compute_output_spec(self, x):
        if None in x.shape:
            output_shape = [
                None,
            ]
        else:
            output_shape = [int(np.prod(x.shape))]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.ravel", "keras_core.ops.numpy.ravel"])
def ravel(x):
    """Return a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Ravel().symbolic_call(x)
    return backend.numpy.ravel(x)


class Real(Operation):
    def call(self, x):
        return backend.numpy.real(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


@keras_core_export(["keras_core.ops.real", "keras_core.ops.numpy.real"])
def real(x):
    """Return the real part of the complex argument.

    Args:
        x: Input tensor.

    Returns:
        The real component of the complex argument.
    """
    if any_symbolic_tensors((x,)):
        return Real().symbolic_call(x)
    return backend.numpy.real(x)


class Reciprocal(Operation):
    def call(self, x):
        return backend.numpy.reciprocal(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


@keras_core_export(
    [
        "keras_core.ops.reciprocal",
        "keras_core.ops.numpy.reciprocal",
    ]
)
def reciprocal(x):
    """Return the reciprocal of the argument, element-wise.

    Calculates `1/x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise reciprocal of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Reciprocal().symbolic_call(x)
    return backend.numpy.reciprocal(x)


class Repeat(Operation):
    def __init__(self, repeats, axis=None):
        super().__init__()
        self.axis = axis
        self.repeats = repeats

    def call(self, x):
        return backend.numpy.repeat(x, self.repeats, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if self.axis is None:
            if None in x_shape:
                return KerasTensor([None], dtype=x.dtype)

            x_flatten_size = int(np.prod(x_shape))
            if isinstance(self.repeats, int):
                output_shape = [x_flatten_size * self.repeats]
            else:
                output_shape = [int(np.sum(self.repeats))]
            return KerasTensor(output_shape, dtype=x.dtype)

        size_on_ax = x_shape[self.axis]
        output_shape = x_shape
        if isinstance(self.repeats, int):
            if size_on_ax is None:
                output_shape[self.axis] = None
            else:
                output_shape[self.axis] = size_on_ax * self.repeats
        else:
            output_shape[self.axis] = int(np.sum(self.repeats))
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.repeat", "keras_core.ops.numpy.repeat"])
def repeat(x, repeats, axis=None):
    """Repeat each element of a tensor after themselves.

    Args:
        x: Input tensor.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values. By default, use
            the flattened input array, and return a flat output array.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Repeat(repeats, axis=axis).symbolic_call(x)
    return backend.numpy.repeat(x, repeats, axis=axis)


class Reshape(Operation):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def call(self, x):
        return backend.numpy.reshape(x, self.new_shape)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_reshape_output_shape(
            x.shape, self.new_shape, "new_shape"
        )
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.reshape", "keras_core.ops.numpy.reshape"])
def reshape(x, new_shape):
    """Gives a new shape to a tensor without changing its data.

    Args:
        x: Input tensor.
        new_shape: The new shape should be compatible with the original shape.
            One shape dimension can be -1 in which case the value is
            inferred from the length of the array and remaining dimensions.

    Returns:
        The reshaped tensor.
    """
    if any_symbolic_tensors((x,)):
        return Reshape(new_shape).symbolic_call(x)
    return backend.numpy.reshape(x, new_shape)


class Roll(Operation):
    def __init__(self, shift, axis=None):
        super().__init__()
        self.shift = shift
        self.axis = axis

    def call(self, x):
        return backend.numpy.roll(x, self.shift, self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.roll", "keras_core.ops.numpy.roll"])
def roll(x, shift, axis=None):
    """Roll tensor elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        x: Input tensor.
        shift: The number of places by which elements are shifted.
        axis: The axis along which elements are shifted. By default, the
            array is flattened before shifting, after which the original
            shape is restored.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Roll(shift, axis=axis).symbolic_call(x)
    return backend.numpy.roll(x, shift, axis=axis)


class Round(Operation):
    def __init__(self, decimals=0):
        super().__init__()
        self.decimals = decimals

    def call(self, x):
        return backend.numpy.round(x, self.decimals)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.round", "keras_core.ops.numpy.round"])
def round(x, decimals=0):
    """Evenly round to the given number of decimals.

    Args:
        x: Input tensor.
        decimals: Number of decimal places to round to. Default is 0.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Round(decimals).symbolic_call(x)
    return backend.numpy.round(x, decimals)


class Sign(Operation):
    def call(self, x):
        return backend.numpy.sign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="int32")


@keras_core_export(["keras_core.ops.sign", "keras_core.ops.numpy.sign"])
def sign(x):
    """Returns a tensor with the signs of the elements of `x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sign().symbolic_call(x)
    return backend.numpy.sign(x)


class Sin(Operation):
    def call(self, x):
        return backend.numpy.sin(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


@keras_core_export(["keras_core.ops.sin", "keras_core.ops.numpy.sin"])
def sin(x):
    """Trigonomeric sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sin().symbolic_call(x)
    return backend.numpy.sin(x)


class Sinh(Operation):
    def call(self, x):
        return backend.numpy.sinh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.sinh", "keras_core.ops.numpy.sinh"])
def sinh(x):
    """Hyperbolic sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sinh().symbolic_call(x)
    return backend.numpy.sinh(x)


class Size(Operation):
    def call(self, x):
        return backend.numpy.size(x)

    def compute_output_spec(self, x):
        return KerasTensor([], dtype="int32")


@keras_core_export(["keras_core.ops.size", "keras_core.ops.numpy.size"])
def size(x):
    """Return the number of elements in a tensor.

    Args:
        x: Input tensor.

    Returns:
        Number of elements in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Size().symbolic_call(x)
    return backend.numpy.size(x)


class Sort(Operation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.sort(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


@keras_core_export(["keras_core.ops.sort", "keras_core.ops.numpy.sort"])
def sort(x, axis=-1):
    """Sorts the elements of `x` along a given axis in ascending order.

    Args:
        x: Input tensor.
        axis: Axis along which to sort. If `None`, the tensor is flattened
            before sorting. Default is the last axis.

    Returns:
        Sorted tensor.
    """
    if any_symbolic_tensors((x,)):
        return Sort(axis=axis).symbolic_call(x)
    return backend.numpy.sort(x, axis=axis)


class Split(Operation):
    def __init__(self, indices_or_sections, axis=0):
        super().__init__()
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def call(self, x):
        return backend.numpy.split(x, self.indices_or_sections, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        x_size_on_axis = x_shape[self.axis]
        if isinstance(self.indices_or_sections, int):
            if x_size_on_axis is None:
                x_shape[self.axis] = None
                return [
                    KerasTensor(x_shape, dtype=x.dtype)
                    for _ in range(self.indices_or_sections)
                ]
            if np.mod(x_size_on_axis, self.indices_or_sections) != 0:
                raise ValueError(
                    "`x` size on given `axis` must be dividible by "
                    "`indices_or_sections` when `indices_or_sections` is an "
                    f"int. But received {x_size_on_axis} and "
                    f"{self.indices_or_sections}."
                )
            size = x_size_on_axis // self.indices_or_sections
            x_shape[self.axis] = size
            return [
                KerasTensor(x_shape, dtype=x.dtype)
                for _ in range(self.indices_or_sections)
            ]

        indices_or_sections = [0] + self.indices_or_sections
        output_size = np.diff(indices_or_sections)
        outputs = []
        for i in range(len(output_size)):
            output_shape = list(x_shape)
            output_shape[self.axis] = int(output_size[i])
            outputs.append(KerasTensor(output_shape, dtype=x.dtype))
        return outputs


@keras_core_export(["keras_core.ops.split", "keras_core.ops.numpy.split"])
def split(x, indices_or_sections, axis=0):
    """Split a tensor into chunks.

    Args:
        x: Input tensor.
        indices_or_sections: Either an integer indicating the number of
            sections along `axis` or a list of integers indicating the indices
            along `axis` at which the tensor is split.
        indices_or_sections: If an integer, N, the tensor will be split into N
            equal sections along `axis`. If a 1-D array of sorted integers,
            the entries indicate indices at which the tensor will be split
            along `axis`.
        axis: Axis along which to split. Default is 0.

    Note:
        A split does not have to result in equal division when using
        Torch backend.

    Returns:
        A list of tensors.
    """
    if any_symbolic_tensors((x,)):
        return Split(indices_or_sections, axis=axis).symbolic_call(x)
    return backend.numpy.split(x, indices_or_sections, axis=axis)


class Stack(Operation):
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def call(self, xs):
        return backend.numpy.stack(xs, axis=self.axis)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape. But found "
                    f"element of shape {x.shape},  which is different from the "
                    f"first element's shape {first_shape}."
                )

        size_on_axis = len(xs)
        output_shape = list(first_shape)
        if self.axis == -1:
            output_shape = output_shape + [size_on_axis]
        elif self.axis >= 0:
            output_shape.insert(self.axis, size_on_axis)
        else:
            output_shape.insert(self.axis + 1, size_on_axis)
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.stack", "keras_core.ops.numpy.stack"])
def stack(x, axis=0):
    """Join a sequence of tensors along a new axis.

    The `axis` parameter specifies the index of the new axis in the
    dimensions of the result.

    Args:
        x: A sequence of tensors.
        axis: Axis along which to stack. Default is 0.

    Returns:
        The stacked tensor.
    """
    if any_symbolic_tensors((x,)):
        return Stack(axis=axis).symbolic_call(x)
    return backend.numpy.stack(x, axis=axis)


class Std(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.std(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
        )


@keras_core_export(["keras_core.ops.std", "keras_core.ops.numpy.std"])
def std(x, axis=None, keepdims=False):
    """Compute the standard deviation along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute standard deviation.
            Default is to compute the standard deviation of the
            flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the standard deviation values.
    """
    if any_symbolic_tensors((x,)):
        return Std(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.std(x, axis=axis, keepdims=keepdims)


class Swapaxes(Operation):
    def __init__(self, axis1, axis2):
        super().__init__()

        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.swapaxes(x, self.axis1, self.axis2)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        tmp = x_shape[self.axis1]
        x_shape[self.axis1] = x_shape[self.axis2]
        x_shape[self.axis2] = tmp
        return KerasTensor(x_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.swapaxes", "keras_core.ops.numpy.swapaxes"])
def swapaxes(x, axis1, axis2):
    """Interchange two axes of a tensor.

    Args:
        x: Input tensor.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        A tensor with the axes swapped.
    """
    if any_symbolic_tensors((x,)):
        return Swapaxes(axis1, axis2).symbolic_call(x)
    return backend.numpy.swapaxes(x, axis1=axis1, axis2=axis2)


class Take(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x, indices):
        return backend.numpy.take(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        x_shape = list(x.shape)
        if isinstance(indices, KerasTensor):
            indices_shape = list(indices.shape)
        else:
            indices_shape = list(getattr(np.array(indices), "shape", []))
        if self.axis is None:
            return KerasTensor(indices_shape, dtype=x.dtype)

        # make sure axis is non-negative
        axis = len(x_shape) + self.axis if self.axis < 0 else self.axis
        output_shape = x_shape[:axis] + indices_shape + x_shape[axis + 1 :]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.take", "keras_core.ops.numpy.take"])
def take(x, indices, axis=None):
    """Take elements from a tensor along an axis.

    Args:
        x: Source tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the
            flattened input tensor is used.

    Returns:
        The corresponding tensor of values.
    """
    if any_symbolic_tensors((x, indices)):
        return Take(axis=axis).symbolic_call(x, indices)
    return backend.numpy.take(x, indices, axis=axis)


class TakeAlongAxis(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x, indices):
        return backend.numpy.take_along_axis(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        x_shape = list(x.shape)
        indices_shape = list(indices.shape)
        if self.axis is None:
            x_shape = [None] if None in x_shape else [int(np.prod(x_shape))]

        if len(x_shape) != len(indices_shape):
            raise ValueError(
                "`x` and `indices` must have the same number of dimensions, "
                f"but receive shape {x_shape} and {indices_shape}."
            )

        del x_shape[self.axis]
        del indices_shape[self.axis]
        output_shape = broadcast_shapes(x_shape, indices_shape)
        size_on_axis = indices.shape[self.axis]
        if self.axis == -1:
            output_shape = output_shape + [size_on_axis]
        elif self.axis >= 0:
            output_shape.insert(self.axis, size_on_axis)
        else:
            output_shape.insert(self.axis + 1, size_on_axis)

        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.take_along_axis",
        "keras_core.ops.numpy.take_along_axis",
    ]
)
def take_along_axis(x, indices, axis=None):
    """Select values from `x` at the 1-D `indices` along the given axis.

    Args:
        x: Source tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the flattened
            input tensor is used.

    Returns:
        The corresponding tensor of values.
    """
    if any_symbolic_tensors((x, indices)):
        return TakeAlongAxis(axis=axis).symbolic_call(x, indices)
    return backend.numpy.take_along_axis(x, indices, axis=axis)


class Tan(Operation):
    def call(self, x):
        return backend.numpy.tan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


@keras_core_export(["keras_core.ops.tan", "keras_core.ops.numpy.tan"])
def tan(x):
    """Compute tangent, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tan().symbolic_call(x)
    return backend.numpy.tan(x)


class Tanh(Operation):
    def call(self, x):
        return backend.numpy.tanh(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.tanh", "keras_core.ops.numpy.tanh"])
def tanh(x):
    """Hyperbolic tangent, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tanh().symbolic_call(x)
    return backend.numpy.tanh(x)


class Tensordot(Operation):
    def __init__(self, axes=2):
        super().__init__()
        self.axes = axes

    def call(self, x1, x2):
        return backend.numpy.tensordot(x1, x2, axes=self.axes)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        if not isinstance(self.axes, int):
            x1_select_shape = [x1_shape[ax] for ax in self.axes[0]]
            x2_select_shape = [x2_shape[ax] for ax in self.axes[1]]
            if not shape_equal(
                x1_select_shape, x2_select_shape, allow_none=True
            ):
                raise ValueError(
                    "Shape mismatch on `x1[axes[0]]` and `x2[axes[1]]`, "
                    f"received {x1_select_shape} and {x2_select_shape}."
                )

            for ax in self.axes[0]:
                x1_shape[ax] = -1
            for ax in self.axes[1]:
                x2_shape[ax] = -1

            x1_shape = list(filter((-1).__ne__, x1_shape))
            x2_shape = list(filter((-1).__ne__, x2_shape))

            output_shape = x1_shape + x2_shape
            return KerasTensor(output_shape, dtype=x1.dtype)

        if self.axes <= 0:
            output_shape = x1_shape + x2_shape
        else:
            output_shape = x1_shape[: -self.axes] + x2_shape[self.axes :]

        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    ["keras_core.ops.tensordot", "keras_core.ops.numpy.tensordot"]
)
def tensordot(x1, x2, axes=2):
    """Compute the tensor dot product along specified axes.

    Args:
        x1: First tensor.
        x2: Second tensor.
        axes: - If an integer, N, sum over the last N axes of `x1` and the
                first N axes of `x2` in order. The sizes of the corresponding
                axes must match.
              - Or, a list of axes to be summed over, first sequence applying
                to `x1`, second to `x2`. Both sequences must be of the
                same length.

    Returns:
        The tensor dot product of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Tensordot(axes=axes).symbolic_call(x1, x2)
    return backend.numpy.tensordot(x1, x2, axes=axes)


class Tile(Operation):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def call(self, x):
        return backend.numpy.tile(x, self.repeats)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        repeats = self.repeats
        if len(x_shape) > len(repeats):
            repeats = [1] * (len(x_shape) - len(repeats)) + repeats
        else:
            x_shape = [1] * (len(repeats) - len(x_shape)) + x_shape

        output_shape = []
        for x_size, repeat in zip(x_shape, repeats):
            if x_size is None:
                output_shape.append(None)
            else:
                output_shape.append(x_size * repeat)
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.tile", "keras_core.ops.numpy.tile"])
def tile(x, repeats):
    """Repeat `x` the number of times given by `repeats`.

    If `repeats` has length `d`, the result will have dimension of
    `max(d, x.ndim)`.

    If `x.ndim < d`, `x` is promoted to be d-dimensional by prepending
    new axes.

    If `x.ndim > d`, `repeats` is promoted to `x.ndim` by prepending 1's to it.

    Args:
        x: Input tensor.
        repeats: The number of repetitions of `x` along each axis.

    Returns:
        The tiled output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Tile(
            repeats,
        ).symbolic_call(x)
    return backend.numpy.tile(x, repeats)


class Trace(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1):
        super().__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.trace(
            x, offset=self.offset, axis1=self.axis1, axis2=self.axis2
        )

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        x_shape[self.axis1] = -1
        x_shape[self.axis2] = -1
        output_shape = list(filter((-1).__ne__, x_shape))
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.trace", "keras_core.ops.numpy.trace"])
def trace(x, offset=0, axis1=0, axis2=1):
    """Return the sum along diagonals of the tensor.

    If `x` is 2-D, the sum along its diagonal with the given offset is
    returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.

    If a has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-arrays whose traces are
    returned.

    The shape of the resulting tensor is the same as that of `x` with `axis1`
    and `axis2` removed.

    Args:
        x: Input tensor.
        offset: Offset of the diagonal from the main diagonal. Can be
            both positive and negative. Defaults to 0.
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
            Defaults to 0 (first axis).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.
            Defaults to 1 (second axis).

    Returns:
        If `x` is 2-D, the sum of the diagonal is returned. If `x` has
        larger dimensions, then a tensor of sums along diagonals is
        returned.
    """
    if any_symbolic_tensors((x,)):
        return Trace(offset, axis1, axis2).symbolic_call(x)
    return backend.numpy.trace(x, offset=offset, axis1=axis1, axis2=axis2)


class Tri(Operation):
    def call(self, N, M=None, k=0, dtype="float32"):
        return backend.numpy.tri(N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype="float32"):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=dtype)


@keras_core_export(["keras_core.ops.tri", "keras_core.ops.numpy.tri"])
def tri(N, M=None, k=0, dtype="float32"):
    """Return a tensor with ones at and below a diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the tensor.
        M: Number of columns in the tensor.
        k: The sub-diagonal at and below which the array is filled.
            `k = 0` is the main diagonal, while `k < 0` is below it, and
            `k > 0` is above. The default is 0.
        dtype: Data type of the returned tensor. The default is "float32".

    Returns:
        Tensor with its lower triangle filled with ones and zeros elsewhere.
        `T[i, j] == 1` for `j <= i + k`, 0 otherwise.
    """
    return backend.numpy.tri(N, M=M, k=k, dtype=dtype)


class Tril(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.tril(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.tril", "keras_core.ops.numpy.tril"])
def tril(x, k=0):
    """Return lower triangle of a tensor.

    For tensors with `ndim` exceeding 2, `tril` will apply to the
    final two axes.

    Args:
        x: Input tensor.
        k: Diagonal above which to zero elements. Defaults to 0, the
            main diagonal. `k < 0` is below it, and `k > 0` is above it.

    Returns:
        Lower triangle of `x`, of same shape and data type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tril(k=k).symbolic_call(x)
    return backend.numpy.tril(x, k=k)


class Triu(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.triu(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.triu", "keras_core.ops.numpy.triu"])
def triu(x, k=0):
    """Return upper triangle of a tensor.

    For tensors with `ndim` exceeding 2, `triu` will apply to the
    final two axes.

    Args:
        x: Input tensor.
        k: Diagonal below which to zero elements. Defaults to 0, the
            main diagonal. `k < 0` is below it, and `k > 0` is above it.

    Returns:
        Upper triangle of `x`, of same shape and data type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Triu(k=k).symbolic_call(x)
    return backend.numpy.triu(x, k=k)


class Vdot(Operation):
    def call(self, x1, x2):
        return backend.numpy.vdot(x1, x2)

    def compute_output_spec(self, x1, x2):
        return KerasTensor([], dtype=x1.dtype)


@keras_core_export(["keras_core.ops.vdot", "keras_core.ops.numpy.vdot"])
def vdot(x1, x2):
    """Return the dot product of two vectors.

    If the first argument is complex, the complex conjugate of the first
    argument is used for the calculation of the dot product.

    Multidimensional tensors are flattened before the dot product is taken.

    Args:
        x1: First input tensor. If complex, its complex conjugate is taken
            before calculation of the dot product.
        x2: Second input tensor.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return Vdot().symbolic_call(x1, x2)
    return backend.numpy.vdot(x1, x2)


class Vstack(Operation):
    def call(self, xs):
        return backend.numpy.vstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[0], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[0] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[0]
        output_shape = list(first_shape)
        output_shape[0] = total_size_on_axis
        return KerasTensor(output_shape)


@keras_core_export(["keras_core.ops.vstack", "keras_core.ops.numpy.vstack"])
def vstack(xs):
    """Stack tensors in sequence vertically (row wise).

    Args:
        xs: Sequence of tensors.

    Returns:
        Tensor formed by stacking the given tensors.
    """
    if any_symbolic_tensors((xs,)):
        return Vstack().symbolic_call(xs)
    return backend.numpy.vstack(xs)


class Where(Operation):
    def call(self, condition, x1, x2):
        return backend.numpy.where(condition, x1, x2)

    def compute_output_spec(self, condition, x1, x2):
        condition_shape = getattr(condition, "shape", [])
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(condition_shape, x1_shape)
        output_shape = broadcast_shapes(output_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.where", "keras_core.ops.numpy.where"])
def where(condition, x1, x2):
    """Return elements chosen from `x1` or `x2` depending on `condition`.

    Args:
        condition: Where `True`, yield `x1`, otherwise yield `x2`.
        x1: Values from which to choose when `condition` is `True`.
        x2: Values from which to choose when `condition` is `False`.

    Returns:
        A tensor with elements from `x1` where `condition` is `True`, and
        elements from `x2` where `condition` is `False`.
    """
    if any_symbolic_tensors((condition, x1, x2)):
        return Where().symbolic_call(condition, x1, x2)
    return backend.numpy.where(condition, x1, x2)


class Subtract(Operation):
    def call(self, x1, x2):
        return backend.numpy.subtract(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.subtract", "keras_core.ops.numpy.subtract"])
def subtract(x1, x2):
    """Subtract arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise difference of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Subtract().symbolic_call(x1, x2)
    return backend.numpy.subtract(x1, x2)


class Multiply(Operation):
    def call(self, x1, x2):
        return backend.numpy.multiply(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.multiply", "keras_core.ops.numpy.multiply"])
def multiply(x1, x2):
    """Multiply arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Multiply().symbolic_call(x1, x2)
    return backend.numpy.multiply(x1, x2)


class Divide(Operation):
    def call(self, x1, x2):
        return backend.numpy.divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.divide", "keras_core.ops.numpy.divide"])
def divide(x1, x2):
    """Divide arguments element-wise.

    `keras_core.ops.true_divide` is an alias for this function.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, the quotient `x1/x2`, element-wise.
    """
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.numpy.divide(x1, x2)


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.numpy.true_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    [
        "keras_core.ops.true_divide",
        "keras_core.ops.numpy.true_divide",
    ]
)
def true_divide(x1, x2):
    """Alias for `keras_core.ops.divide`."""
    if any_symbolic_tensors((x1, x2)):
        return TrueDivide().symbolic_call(x1, x2)
    return backend.numpy.true_divide(x1, x2)


class Power(Operation):
    def call(self, x1, x2):
        return backend.numpy.power(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(["keras_core.ops.power", "keras_core.ops.numpy.power"])
def power(x1, x2):
    """First tensor elements raised to powers from second tensor, element-wise.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        Output tensor, the bases in `x1` raised to the exponents in `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Power().symbolic_call(x1, x2)
    return backend.numpy.power(x1, x2)


class Negative(Operation):
    def call(self, x):
        return backend.numpy.negative(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.negative", "keras_core.ops.numpy.negative"])
def negative(x):
    """Numerical negative, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, `y = -x`.
    """
    if any_symbolic_tensors((x,)):
        return Negative().symbolic_call(x)
    return backend.numpy.negative(x)


class Square(Operation):
    def call(self, x):
        return backend.numpy.square(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.square", "keras_core.ops.numpy.square"])
def square(x):
    """Return the element-wise square of the input.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, the square of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.numpy.square(x)


class Sqrt(Operation):
    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.numpy.sqrt(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.sqrt", "keras_core.ops.numpy.sqrt"])
def sqrt(x):
    """Return the non-negative square root of a tensor, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, the non-negative square root of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sqrt().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return backend.numpy.sqrt(x)


class Squeeze(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.squeeze(x, axis=self.axis)

    def compute_output_spec(self, x, axis=None):
        input_shape = list(x.shape)
        if axis is None:
            output_shape = list(filter((1).__ne__, input_shape))
            return KerasTensor(output_shape)
        else:
            if input_shape[axis] != 1:
                raise ValueError(
                    f"Cannot squeeze axis {axis}, because the dimension is not "
                    "1."
                )
            del input_shape[axis]
            return KerasTensor(input_shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.squeeze", "keras_core.ops.numpy.squeeze"])
def squeeze(x, axis=None):
    """Remove axes of length one from `x`.

    Args:
        x: Input tensor.
        axis: Select a subset of the entries of length one in the shape.

    Returns:
        The input tensor with all or a subset of the dimensions of
        length 1 removed.
    """
    if any_symbolic_tensors((x,)):
        return Squeeze().symbolic_call(x, axis=axis)
    return backend.numpy.squeeze(x, axis=axis)


class Transpose(Operation):
    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes

    def call(self, x):
        return backend.numpy.transpose(x, axes=self.axes)

    def compute_output_spec(self, x):
        x_shape = x.shape
        if self.axes is None:
            return KerasTensor(x_shape[::-1])

        if len(self.axes) != len(x_shape):
            raise ValueError(
                "axis must be a list of the same length as the input shape, "
                f"expected {len(x_shape)}, but received {len(self.axes)}."
            )
        output_shape = []
        for ax in self.axes:
            output_shape.append(x_shape[ax])
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_core_export(
    ["keras_core.ops.transpose", "keras_core.ops.numpy.transpose"]
)
def transpose(x, axes=None):
    """Returns a tensor with `axes` transposed.

    Args:
        x: Input tensor.
        axes: Sequence of integers. Permutation of the dimensions of `x`.
            By default, the order of the axes are reversed.

    Returns:
        `x` with its axes permuted.
    """
    if any_symbolic_tensors((x,)):
        return Transpose(axes=axes).symbolic_call(x)
    return backend.numpy.transpose(x, axes=axes)


class Mean(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.mean(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.mean", "keras_core.ops.numpy.mean"])
def mean(x, axis=None, keepdims=False):
    """Compute the arithmetic mean along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the means are computed. The default
            is to compute the mean of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the mean values.
    """
    if any_symbolic_tensors((x,)):
        return Mean(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.mean(x, axis=axis, keepdims=keepdims)


class Var(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.var(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.var", "keras_core.ops.numpy.var"])
def var(x, axis=None, keepdims=False):
    """Compute the variance along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the variance is computed. The default
            is to compute the variance of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the variance.
    """
    if any_symbolic_tensors((x,)):
        return Var(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.var(x, axis=axis, keepdims=keepdims)


class Sum(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.sum(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_core_export(["keras_core.ops.sum", "keras_core.ops.numpy.sum"])
def sum(x, axis=None, keepdims=False):
    """Sum of a tensor over the given axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the sum is computed. The default is to
            compute the sum of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the sum.
    """
    if any_symbolic_tensors((x,)):
        return Sum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.sum(x, axis=axis, keepdims=keepdims)


class Zeros(Operation):
    def call(self, shape, dtype="float32"):
        return backend.numpy.zeros(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


@keras_core_export(["keras_core.ops.zeros", "keras_core.ops.numpy.zeros"])
def zeros(shape, dtype="float32"):
    """Return a new tensor of given shape and type, filled with zeros.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.

    Returns:
        Tensor of zeros with the given shape and dtype.
    """
    return backend.numpy.zeros(shape, dtype=dtype)


class Ones(Operation):
    def call(self, shape, dtype="float32"):
        return backend.numpy.ones(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


@keras_core_export(["keras_core.ops.ones", "keras_core.ops.numpy.ones"])
def ones(shape, dtype="float32"):
    """Return a new tensor of given shape and type, filled with ones.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.

    Returns:
        Tensor of ones with the given shape and dtype.
    """
    return backend.numpy.ones(shape, dtype=dtype)


class Eye(Operation):
    def call(self, N, M=None, k=0, dtype="float32"):
        return backend.numpy.eye(N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype="float32"):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=dtype)


@keras_core_export(["keras_core.ops.eye", "keras_core.ops.numpy.eye"])
def eye(N, M=None, k=0, dtype="float32"):
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output.
        M: Number of columns in the output. If `None`, defaults to `N`.
        k: Index of the diagonal: 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal,
            and a negative value to a lower diagonal.
        dtype: Data type of the returned tensor.

    Returns:
        Tensor with ones on the k-th diagonal and zeros elsewhere.
    """
    return backend.numpy.eye(N, M=M, k=k, dtype=dtype)


class FloorDivide(Operation):
    def call(self, x1, x2):
        return backend.numpy.floor_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    ["keras_core.ops.floor_divide", "keras_core.ops.numpy.floor_divide"]
)
def floor_divide(x1, x2):
    """Returns the largest integer smaller or equal to the division of inputs.

    Args:
        x1: Numerator.
        x2: Denominator.

    Returns:
        Output tensor, `y = floor(x1/x2)`
    """
    if any_symbolic_tensors((x1, x2)):
        return FloorDivide().symbolic_call(x1, x2)
    return backend.numpy.floor_divide(x1, x2)


class LogicalXor(Operation):
    def call(self, x1, x2):
        return backend.numpy.logical_xor(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


@keras_core_export(
    ["keras_core.ops.logical_xor", "keras_core.ops.numpy.logical_xor"]
)
def logical_xor(x1, x2):
    """Compute the truth value of `x1 XOR x2`, element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalXor().symbolic_call(x1, x2)
    return backend.numpy.logical_xor(x1, x2)
