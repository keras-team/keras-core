"""
segment_sum
top_k
in_top_k
logsumexp
"""

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops.operation import Operation
from keras_core.ops.operation_utils import reduce_shape


class SegmentSum(Operation):
    def __init__(self, num_segments=None, sorted=False):
        super().__init__()
        self.num_segments = num_segments
        self.sorted = sorted

    def compute_output_spec(self, data, segment_ids):
        num_segments = self.num_segments
        output_shape = (num_segments,) + tuple(data.shape[1:])
        return KerasTensor(shape=output_shape, dtype=data.dtype)

    def call(self, data, segment_ids):
        return backend.math.segment_sum(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_core_export("keras_core.ops.segment_sum")
def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if any_symbolic_tensors((data,)):
        return SegmentSum(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.math.segment_sum(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class TopK(Operation):
    def __init__(self, k, sorted=False):
        super().__init__()
        self.k = k
        self.sorted = sorted

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        output_shape[-1] = self.k
        # Return a tuple (values, indices).
        return (
            KerasTensor(shape=output_shape, dtype=x.dtype),
            KerasTensor(shape=output_shape, dtype="int32"),
        )

    def call(self, x):
        return backend.math.top_k(x, self.k, self.sorted)


@keras_core_export("keras_core.ops.top_k")
def top_k(x, k, sorted=True):
    if any_symbolic_tensors((x,)):
        return TopK(k, sorted).symbolic_call(x)
    return backend.math.top_k(x, k, sorted)


class InTopK(Operation):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def compute_output_spec(self, targets, predictions):
        return KerasTensor(shape=targets.shape, dtype="bool")

    def call(self, targets, predictions):
        return backend.math.in_top_k(targets, predictions, self.k)


@keras_core_export("keras_core.ops.in_top_k")
def in_top_k(targets, predictions, k):
    if any_symbolic_tensors((targets, predictions)):
        return InTopK(k).symbolic_call(targets, predictions)
    return backend.math.in_top_k(targets, predictions, k)


class Logsumexp(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def compute_output_spec(self, x):
        output_shape = reduce_shape(x.shape, self.axis, self.keepdims)
        return KerasTensor(shape=output_shape)

    def call(self, x):
        return backend.math.logsumexp(x, axis=self.axis, keepdims=self.keepdims)


@keras_core_export("keras_core.ops.logsumexp")
def logsumexp(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Logsumexp(axis, keepdims).symbolic_call(x)
    return backend.math.logsumexp(x, axis=axis, keepdims=keepdims)


class Qr(Operation):
    def __init__(self, mode="reduced"):
        super().__init__()
        if mode not in {"reduced", "complete"}:
            raise ValueError(
                "`mode` argument value not supported. "
                "Expected one of {'reduced', 'complete'}. "
                f"Received: mode={mode}"
            )
        self.mode = mode

    def compute_output_spec(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                "Input should have rank >= 2. Received: "
                f"input.shape = {x.shape}"
            )
        m = x.shape[-2]
        n = x.shape[-1]
        if m is None or n is None:
            raise ValueError(
                "Input should have its last 2 dimensions "
                "fully-defined. Received: "
                f"input.shape = {x.shape}"
            )
        k = min(m, n)
        base = tuple(x.shape[:-2])
        if self.mode == "reduced":
            return (
                KerasTensor(shape=base + (m, k), dtype=x.dtype),
                KerasTensor(shape=base + (k, n), dtype=x.dtype),
            )
        # 'complete' mode.
        return (
            KerasTensor(shape=base + (m, m), dtype=x.dtype),
            KerasTensor(shape=base + (m, n), dtype=x.dtype),
        )

    def call(self, x):
        return backend.math.qr(x, mode=self.mode)


@keras_core_export("keras_core.ops.qr")
def qr(x, mode="reduced"):
    if any_symbolic_tensors((x,)):
        return Qr(mode=mode).symbolic_call(x)
    return backend.math.qr(x, mode=mode)


class FFT(Operation):
    def __init__(self, n=None, axis=-1, norm=None):
        super().__init__()
        if norm is not None and norm not in ["backward", "ortho", "forward"]:
            raise ValueError(
                "`norm` argument value not supported. "
                'Expected one of `{None, "backward", "ortho", "forward"}`. '
                f"Received: norm={norm}"
            )
        self.n = n
        self.axis = axis
        self.norm = norm

    def compute_output_spec(self, a):
        real, imag = a

        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and imaginary."
                "Both real and imaginary should have the same shape. "
                f"Received: real.shape = {real.shape}, imag.shape = {imag.shape}"
            )

        # We are calculating 1D FFT. Hence, rank >= 1.
        if len(real.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {real.shape}"
            )

        # axis should be within bounds.
        if self.axis is not None and (self.axis < -len(real.shape) or self.axis >= len(real.shape)):
            raise ValueError(
                f"Out-of-bounds axis {self.axis}. "
                f"Received: input.shape = {real.shape} with axis = {self.axis}"
            )

        # The axis along which we are calculating FFT should be fully-defined.
        m = real.shape[self.axis]
        if m is None:
            raise ValueError(
                f"Input should have its {self.axis}th axis fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        output_shape = real.shape
        output_shape[self.axis] = self.n if self.n is not None else m

        return (
            KerasTensor(shape=output_shape, dtype=real.dtype),
            KerasTensor(shape=output_shape, dtype=real.dtype),
        )

    def call(self, x):
        return backend.math.fft(x)


class FFT2(Operation):
    def __init__(self, s=None, axes=(-2, -1), norm=None):
        super().__init__()
        if s is not None:
            if not isinstance(s, tuple) or len(s) != 2:
                raise ValueError(
                    f"`s` should be either be `None` or a tuple of two integers. "
                    f"Received: s={s}"
                )
        if not isinstance(axes, tuple) or len(axes) != 2:
            raise ValueError(
                f"`axes` should be a tuple of two integers. "
                f"Received: axes={axes}"
            )
        if norm is not None and norm not in ["backward", "ortho", "forward"]:
            raise ValueError(
                "`norm` argument value not supported. "
                'Expected one of `{None, "backward", "ortho", "forward"}`. '
                f"Received: norm={norm}"
            )
        self.s = s
        self.axes = axes
        self.norm = norm

    def compute_output_spec(self, a):
        real, imag = a
        
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and imaginary."
                "Both real and imaginary should have the same shape. "
                f"Received: real.shape = {real.shape}, imag.shape = {imag.shape}"
            )
        # We are calculating 2D FFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes should be within bounds.
        for axis in self.axes:
            if axis is not None and (axis < -len(real.shape) or axis >= len(real.shape)):
                raise ValueError(
                    f"Out-of-bounds axes index {self.axes}. "
                    f"Received: input.shape = {real.shape} with axes = {self.axes}"
                )

        # The axes along which we are calculating FFT should be fully-defined.
        m = real.shape[self.axes[0]]
        n = real.shape[self.axes[1]]
        if m is None or n is None:
            raise ValueError(
                f"Input should have its {self.axes} axes fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        output_shape = real.shape
        output_shape[self.axes[0]] = self.s[0] if self.s is not None else m
        output_shape[self.axes[1]] = self.s[1] if self.s is not None else n

        return (
            KerasTensor(shape=output_shape, dtype=real.dtype),
            KerasTensor(shape=output_shape, dtype=real.dtype),
        )

    def call(self, x):
        return backend.math.fft2(x)

class FFTN(Operation):
    def __init__(self, s=None, axes=(-2, -1), norm=None):
        super().__init__()
        if s is not None:
            if not isinstance(s, tuple) or len(s) != 2:
                raise ValueError(
                    f"`s` should be either be `None` or a tuple of two integers. "
                    f"Received: s={s}"
                )
        if not isinstance(axes, tuple) or len(axes) != 2:
            raise ValueError(
                f"`axes` should be a tuple of two integers. "
                f"Received: axes={axes}"
            )
        if norm is not None and norm not in ["backward", "ortho", "forward"]:
            raise ValueError(
                "`norm` argument value not supported. "
                'Expected one of `{None, "backward", "ortho", "forward"}`. '
                f"Received: norm={norm}"
            )
        self.s = s
        self.axes = axes
        self.norm = norm

    def compute_output_spec(self, a):
        real, imag = a
        
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and imaginary."
                "Both real and imaginary should have the same shape. "
                f"Received: real.shape = {real.shape}, imag.shape = {imag.shape}"
            )
        # We are calculating 2D FFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes should be within bounds.
        for axis in self.axes:
            if axis is not None and (axis < -len(real.shape) or axis >= len(real.shape)):
                raise ValueError(
                    f"Out-of-bounds axes index {self.axes}. "
                    f"Received: input.shape = {real.shape} with axes = {self.axes}"
                )

        # The axes along which we are calculating FFT should be fully-defined.
        m = real.shape[self.axes[0]]
        n = real.shape[self.axes[1]]
        if m is None or n is None:
            raise ValueError(
                f"Input should have its {self.axes} axes fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        output_shape = real.shape
        output_shape[self.axes[0]] = self.s[0] if self.s is not None else m
        output_shape[self.axes[1]] = self.s[1] if self.s is not None else n

        return (
            KerasTensor(shape=output_shape, dtype=real.dtype),
            KerasTensor(shape=output_shape, dtype=real.dtype),
        )

    def call(self, x):
        return backend.math.fft2(x)


@keras_core_export("keras_core.ops.fft")
def fft(x):
    if any_symbolic_tensors((x,)):
        return FFT(n=1).symbolic_call(x)
    return backend.math.fft(x)


@keras_core_export("keras_core.ops.fft2")
def fft2(x):
    if any_symbolic_tensors((x,)):
        return FFT(n=2).symbolic_call(x)
    return backend.math.fft2d(x)


@keras_core_export("keras_core.ops.fftn")
def fftn(x):
    if any_symbolic_tensors((x,)):
        return FFT(n=3).symbolic_call(x)
    return backend.math.fft3d(x)
