import torch
from keras_core.backend.torch.core import to_torch_dtype


def _make_torch_val(inputs):
    # This helper converts individual values (int, bool, etc.)
    # to a individual torch tensor for compatibility.
    # All numpy.ndarrays still need to be passed
    # as Torch Tensors.
    if isinstance(inputs, torch.Tensor):
        return inputs
    if isinstance(inputs, (list, tuple)):
        # list/tuple of individual values
        return tuple(
            [torch.tensor(val) if not isinstance(val, torch.Tensor)  else val for val in inputs]
        )
    return torch.tensor(inputs)  # individual value



def add(x1, x2):
    return torch.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    return torch.einsum(subscripts, *operands)


def subtract(x1, x2):
    return torch.subtract(x1, x2)


def matmul(x1, x2):
    return torch.matmul(x1, x2)


def multiply(x1, x2):
    return torch.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    return torch.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    if axis is None:
        result = torch.max(x)
    else:
        result = torch.max(x, dim=axis, keepdim=keepdims)

    if initial is not None:
        return torch.maximum(
            result, initial
        )
    return result


def ones(shape, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    torch.ones(*shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    torch.zeros(*shape, dtype=dtype)


def absolute(x):
    return abs(x)


def abs(x):
    return torch.abs(x)


def all(x, axis=None, keepdims=False):
    if axis is None:
        torch.all(x, dim=axis, keepdim=keepdims)
    else:
        torch.all(x)


def any(x, axis=None, keepdims=False):
    if axis is None:
        torch.any(x, dim=axis, keepdim=keepdims)
    else:
        torch.any(x)


def amax(x, axis=None, keepdims=False):
    if axis is None:
        torch.amax(x, dim=axis, keepdim=keepdims)
    else:
        torch.amax(x)


def amin(x, axis=None, keepdims=False):
    if axis is None:
        torch.amax(x, dim=axis, keepdim=keepdims)
    else:
        torch.amax(x)


def append(
    x1,
    x2,
    axis=None,
):
    dim = axis or 0
    return torch.cat((x1, x2), dim=dim)


def arange(start, stop=None, step=None, dtype=None):
    dtype = to_torch_dtype(dtype)
    if stop is None:
        return torch.arange(start, step=step, dtype=dtype)
    step = step or 1
    return torch.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    return torch.arccos(x)


def arcsin(x):
    return torch.arcsin(x)


def arctan(x):
    return torch.arctan(x)


def arctan2(x1, x2):
    return torch.arctan2(x1, x2)


def argmax(x, axis=None):
    return torch.argmax(x, dim=axis)


def argmin(x, axis=None):
    return torch.argmin(x, dim=axis)


def argsort(x, axis=-1):
    return torch.argsort(x, dim=axis)


def array(x, dtype=None):
    dtype = to_torch_dtype(dtype)
    pass
    # return tfnp.array(x, dtype=dtype)


def average(x, axis=None, weights=None):
    if weights is not None:
        return torch.matmul(x, weights) / torch.sum(weights)
    return torch.mean(x, axis)


def broadcast_to(x, shape):
    return torch.broadcast_to(x, shape)


def ceil(x):
    return torch.ceil(x)


def clip(x, x_min, x_max):
    return torch.clip(x, min=x_min, max=x_max)


def concatenate(xs, axis=0):
    return torch.cat(xs, dim=axis)


def conjugate(x):
    return conj(x)


def conj(x):
    return torch.conj(x)


def copy(x):
    return torch.clone(x)


def cos(x):
    return torch.cos(x)


def count_nonzero(x, axis=None):
    return torch.count_nonzero(x, dim=axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    # There is API divergence between np.cross and torch.cross
    # https://github.com/pytorch/pytorch/issues/50273
    return torch.cross(x1, x2, dim=axis)


def cumprod(x, axis=None):
    return torch.cumprod(x, dim=axis)


def cumsum(x, axis=None):
    return torch.cumsum(x, dim=axis)


def diag(x, k=0):
    return torch.diag(x, diagonal=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return torch.diagonal(
       x,
       offset=offset,
       dim1=axis1,
       dim2=axis2,
    )


def dot(x, y):
    return torch.dot(x, y)


def empty(shape, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    return torch.empty(size=shape, dtype=dtype)


def equal(x1, x2):
    return torch.equal(x1, x2)


def exp(x):
    return torch.exp(x)


def expand_dims(x, axis):
    return torch.unsqueeze(x, dim=axis)


def expm1(x):
    return torch.expm1(x)


def flip(x, axis=None):
    return torch.flip(x, dims=axis)


def floor(x):
    return torch.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = to_torch_dtype(dtype)
    return torch.full(size=shape, fill_value=fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    dtype = to_torch_dtype(dtype)
    return torch.full_like(input=x, fill_value=fill_value, dtype=dtype)


def greater(x1, x2):
    return torch.greater(x1, x2)


def greater_equal(x1, x2):
    return torch.greater_equal(x1, x2)


def hstack(xs):
    return torch.hstack(xs)


def identity(n, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    pass
    # return tfnp.identity(n, dtype=dtype)


def imag(x):
    return torch.imag(x)


def isclose(x1, x2):
    return torch.isclose(x1, x2)


def isfinite(x):
    return torch.isfinite(x)


def isinf(x):
    return torch.isinf(x)


def isnan(x):
    return torch.isnan(x)


def less(x1, x2):
    return torch.less(x1, x2)


def less_equal(x1, x2):
    return torch.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if axis != 0:
        raise ValueError(
        "torch.linspace does not support an `axis` argument. "
        f"Received axis={axis}"
        )
        dtype = to_torch_dtype(dtype)
    steps = abs(stop - start)/num
    linspace = torch.linspace(
       start=start,
       end=stop,
       steps=steps,
       dtype=dtype,
    )
    if endpoint is False:
        return linspace[:-1]
    if retstep is True:
        return (linspace, steps)
    return linspace


def log(x):
    return torch.log(x)


def log10(x):
    return torch.log10(x)


def log1p(x):
    return torch.log1p(x)


def log2(x):
    return torch.log2(x)


def logaddexp(x1, x2):
    return torch.logaddexp(x1, x2)


def logical_and(x1, x2):
    return torch.logical_and(x1, x2)


def logical_not(x):
    return torch.logical_not(x)


def logical_or(x1, x2):
    return torch.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if axis != 0:
        raise ValueError(
        "torch.logspace does not support an `axis` argument. "
        f"Received axis={axis}"
        )
    dtype = to_torch_dtype(dtype)
    steps = abs(stop - start)/num
    logspace = torch.logspace(
       start=start,
       end=stop,
       steps=steps,
       base=base,
       dtype=dtype,
    )
    if endpoint is False:
        return logspace[:-1]
    return logspace


def maximum(x1, x2):
    return torch.maximum(x1, x2)


def meshgrid(*x, indexing="xy"):
    return torch.meshgrid(x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    if axis is None:
        result = torch.min(x)
    else:
        result = torch.min(x, dim=axis, keepdim=keepdims)

    if initial is not None:
        return torch.minimum(
            result, initial
        )
    return result


def minimum(x1, x2):
    return torch.minimum(x1, x2)


def mod(x1, x2):
    return torch.remainder(x1, x2)


def moveaxis(x, source, destination):
    return torch.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    return torch.nan_to_num(x)


def ndim(x):
    return x.ndim


def nonzero(x):
    return torch.nonzero(x)


def not_equal(x1, x2):
    return torch.not_equal(x1, x2)


def ones_like(x, dtype=None):
    dtype = to_torch_dtype(dtype)
    return torch.ones_like(x, dtype=dtype)


def outer(x1, x2):
    return torch.outer(x1, x2)


def pad(x, pad_width, mode="constant"):
    return torch.nn.functional.pad(x, pad=tuple(pad_width), mode=mode)


def prod(x, axis=None, keepdims=False, dtype=None):
    dtype = to_torch_dtype(dtype)
    if axis is None:
        return torch.prod(x, dtype=dtype)
    return torch.prod(x, dim=axis, keepdim=keepdims, dtype=dtype)


def ravel(x):
    return torch.ravel(x)


def real(x):
    return torch.real(x)


def reciprocal(x):
    return torch.reciprocal(x)


def repeat(x, repeats, axis=None):
    return torch.repeat_interleave(x, repeats, dim=axis)


def reshape(x, new_shape):
    return torch.reshape(x, new_shape)


def roll(x, shift, axis=None):
    return torch.roll(x, shift, dims=axis)


def sign(x):
    return torch.sign(x)


def sin(x):
    return torch.sin(x)


def size(x):
    return x.size()


def sort(x, axis=-1):
    return torch.sort(x, dim=axis)


def split(x, indices_or_sections, axis=0):
    return torch.split(
        tensor=x,
        split_size_or_sections=indices_or_sections,
        dim=axis,
    )


def stack(x, axis=0):
    return torch.stack(x, dim=axis)


def std(x, axis=None, keepdims=False):
    return torch.std(x, dim=axis, keepdim=keepdims)


def swapaxes(x, axis1, axis2):
    return torch.swapaxes(x, axis0=axis1, axis1=axis2)


def take(x, indices, axis=None):
    return torch.index_select(x, index=indices, dim=axis)


def take_along_axis(x, indices, axis=None):
    return torch.take_along_dim(x, indices, dim=axis)


def tan(x):
    return torch.tan(x)


def tensordot(x1, x2, axes=2):
    return torch.tensordot(x1, x2, dims=axes)


def round(x, decimals=0):
    return torch.round(x, decimals=decimals)


def tile(x, repeats):
    return torch.tile(x, dims=repeats)


def trace(x, offset=None, axis1=None, axis2=None):
    if offset or axis1 or axis2:
        "Arguments not supported by `torch.trace: "
        f""
    return torch.trace(x)


def tri(N, M=None, k=0, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    pass
    # return tfnp.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    return torch.tril(x, diagonal=k)


def triu(x, k=0):
    return torch.triu(x, diagonal=k)


def vdot(x1, x2):
    return torch.vdot(x1, x2)


def vstack(xs):
    return torch.vstack(xs)


def where(condition, x1, x2):
    return torch.where(condition, x1, x2)


def divide(x1, x2):
    return torch.divide(x1, x2)


def true_divide(x1, x2):
    return torch.true_divide(x1, x2)


def power(x1, x2):
    return torch.pow(x1, x2)


def negative(x):
    return torch.negative(x)


def square(x):
    return torch.square(x)


def sqrt(x):
    return torch.sqrt(x)


def squeeze(x, axis=None):
    return torch.squeeze(x, dim=axis)


def transpose(x, axes=None):
    if axes is not None:
        return torch.permute(x, dims=axes)
    return torch.t(x)


def var(x, axis=None, keepdims=False):
    return torch.var(x, dim=axis, keepdim=keepdims)


def sum(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.sum(x, axis=axis, keepdim=keepdims)
    return torch.sum(x)


def eye(N, M=None, k=None, dtype="float32"):
    if k is not None:
        raise ValueError(
            "Argument not supported by "
            f"`torch.eye`: k={k}"
        )
    dtype = to_torch_dtype(dtype)
    return torch.eye(n=N, m=M, dtype=dtype)
