"""Commonly used math operations not included in NumPy."""

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
    """Computes the sum of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A 1-D tensor containing segment indices for each
            element in `data`.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Default is `False`.

    Returns:
        A tensor containing the sum of segments, where each element
        represents the sum of the corresponding segment in `data`.

    Example:

    >>> data = keras_core.ops.convert_to_tensor([1,2,10,20,100,200])
    >>> segment_ids = keras_core.ops.convert_to_tensor([0,0,1,1,2,2])
    >>> num_segments = 3
    >>> keras_core.ops.segment_sum(data, segment_ids,num_segments)
    Array([  3,  30, 300], dtype=int32)
    """
    
    if any_symbolic_tensors((data,)):
        return SegmentSum(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.math.segment_sum(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class SegmentMax(Operation):
    def __init__(self, num_segments=None, sorted=False):
        super().__init__()
        self.num_segments = num_segments
        self.sorted = sorted

    def compute_output_spec(self, data, segment_ids):
        num_segments = self.num_segments
        output_shape = (num_segments,) + tuple(data.shape[1:])
        return KerasTensor(shape=output_shape, dtype=data.dtype)

    def call(self, data, segment_ids):
        return backend.math.segment_max(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_core_export("keras_core.ops.segment_max")
def segment_max(data, segment_ids, num_segments=None, sorted=False):
    """Computes the max of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A 1-D tensor containing segment indices for each
            element in `data`.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Default is `False`.

    Returns:
        A tensor containing the max of segments, where each element
        represents the max of the corresponding segment in `data`.

    Example:

    >>> data = keras_core.ops.convert_to_tensor([1,2,10,20,100,200])
    >>> segment_ids = keras_core.ops.convert_to_tensor([0,0,1,1,2,2])
    >>> num_segments = 3
    >>> keras_core.ops.segment_max(data, segment_ids,num_segments)
    Array([  2,  20, 200], dtype=int32)
 
    """
    if any_symbolic_tensors((data,)):
        return SegmentMax(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.math.segment_max(
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
    """Finds the top-k values and their indices in a tensor.

    Args:
        x: Input tensor.
        k: An integer representing the number of top elements to retrieve.
        sorted: A boolean indicating whether to sort the output in
        descending order. Default is `True`.

    Returns:
        A tuple containing two tensors. The first tensor contains the
        top-k values, and the second tensor contains the indices of the
        top-k values in the input tensor.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])
    >>> values, indices = top_k(x, k=3)
    >>> print(values)
    array([9 7 5], shape=(3,), dtype=int32)
    >>> print(indices)
    array([4 2 0], shape=(3,), dtype=int32)

    """
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
    """Checks if the targets are in the top-k predictions.

    Args:
        targets: A tensor of true labels.
        predictions: A tensor of predicted labels.
        k: An integer representing the number of predictions to consider.

    Returns:
        A boolean tensor of the same shape as `targets`, where each element
        indicates whether the corresponding target is in the top-k predictions.

    Example:

    >>> targets = keras_core.ops.convert_to_tensor([2, 5, 3])
    >>> predictions = keras_core.ops.convert_to_tensor(
    ... [[0.1, 0.4, 0.6, 0.9, 0.5],
    ...  [0.1, 0.7, 0.9, 0.8, 0.3],
    ...  [0.1, 0.6, 0.9, 0.9, 0.5]])
    >>> in_top_k(targets, predictions, k=3)
    array([ True False  True], shape=(3,), dtype=bool)
    """
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
    """Computes the logarithm of sum of exponentials of elements in a tensor.

    Args:
        x: Input tensor.
        axis: An integer or a tuple of integers specifying the axis/axes
            along which to compute the sum. If `None`, the sum is computed
            over all elements. Default is `None`.
        keepdims: A boolean indicating whether to keep the dimensions of
            the input tensor when computing the sum. Default is `False`.

    Returns:
        A tensor containing the logarithm of the sum of exponentials of
        elements in `x`.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([1., 2., 3.])
    >>> logsumexp(x)
    3.407606
    """
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
    """Computes the QR decomposition of a tensor.

    Args:
        x: Input tensor.
        mode: A string specifying the mode of the QR decomposition.
            - 'reduced': Returns the reduced QR decomposition. (default)
            - 'complete': Returns the complete QR decomposition.

    Returns:
        A tuple containing two tensors. The first tensor represents the
        orthogonal matrix Q, and the second tensor represents the upper
        triangular matrix R.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> q, r = qr(x)
    >>> print(q)
    array([[-0.16903079  0.897085]
           [-0.5070925   0.2760267 ]
           [-0.8451542  -0.34503305]], shape=(3, 2), dtype=float32)
    """

    if any_symbolic_tensors((x,)):
        return Qr(mode=mode).symbolic_call(x)
    return backend.math.qr(x, mode=mode)


class ExtractSequences(Operation):
    def __init__(self, sequence_length, sequence_stride):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {x.shape}"
            )
        if x.shape[-1] is not None:
            num_sequences = (
                1 + (x.shape[-1] - self.sequence_length) // self.sequence_stride
            )
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.sequence_length)
        return KerasTensor(shape=new_shape, dtype=x.dtype)

    def call(self, x):
        return backend.math.extract_sequences(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
        )


@keras_core_export("keras_core.ops.extract_sequences")
def extract_sequences(x, sequence_length, sequence_stride):
    """Expands the dimension of last axis into sequences of `sequence_length`.

    Slides a window of size `sequence_length` over the last axis of the input
    with a stride of `sequence_stride`, replacing the last axis with
    `[num_sequences, sequence_length]` sequences.

    If the dimension along the last axis is N, the number of sequences can be
    computed by:

    `num_sequences = 1 + (N - sequence_length) // sequence_stride`

    Args:
        x: Input tensor.
        sequence_length: An integer representing the sequences length.
        sequence_stride: An integer representing the sequences hop size.

    Returns:
        A tensor of sequences with shape [..., num_sequences, sequence_length].

    Example:

    >>> x = keras_core.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> extract_sequences(x, 3, 2)
    array([[1, 2, 3],
       [3, 4, 5]])
    """
    if any_symbolic_tensors((x,)):
        return ExtractSequences(sequence_length, sequence_stride).symbolic_call(
            x
        )
    return backend.math.extract_sequences(x, sequence_length, sequence_stride)


class FFT(Operation):
    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )

        # We are calculating 1D FFT. Hence, rank >= 1.
        if len(real.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {real.shape}"
            )

        # The axis along which we are calculating FFT should be fully-defined.
        m = real.shape[-1]
        if m is None:
            raise ValueError(
                f"Input should have its {self.axis}th axis fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.math.fft(x)


class FFT2(Operation):
    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        # We are calculating 2D FFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes along which we are calculating FFT should be fully-defined.
        m = real.shape[-1]
        n = real.shape[-2]
        if m is None or n is None:
            raise ValueError(
                f"Input should have its {self.axes} axes fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.math.fft2(x)


@keras_core_export("keras_core.ops.fft")
def fft(x):
    """Computes the Fast Fourier Transform along last axis of input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output tensor.

    Example:

    >>> x = (
    ...     keras_core.ops.convert_to_tensor([1., 2.]),
    ...     keras_core.ops.convert_to_tensor([0., 1.]),
    ... )
    >>> fft(x)
    (array([ 3., -1.], dtype=float32), array([ 1., -1.], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return FFT().symbolic_call(x)
    return backend.math.fft(x)


@keras_core_export("keras_core.ops.fft2")
def fft2(x):
    """Computes the 2D Fast Fourier Transform along the last two axes of input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Example:

    >>> x = (
    ...     keras_core.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
    ...     keras_core.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
    ... )
    >>> fft2(x)
    (array([[ 6.,  0.],
        [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
        [ 0., -2.]], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return FFT2().symbolic_call(x)
    return backend.math.fft2(x)


class RFFT(Operation):
    def __init__(self, fft_length=None):
        super().__init__()
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        # We are calculating 1D RFFT. Hence, rank >= 1.
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {x.shape}"
            )

        if self.fft_length is not None:
            new_last_dimension = self.fft_length // 2 + 1
        else:
            new_last_dimension = x.shape[-1] // 2 + 1
        new_shape = x.shape[:-1] + (new_last_dimension,)

        return (
            KerasTensor(shape=new_shape, dtype=x.dtype),
            KerasTensor(shape=new_shape, dtype=x.dtype),
        )

    def call(self, x):
        return backend.math.rfft(x, fft_length=self.fft_length)


@keras_core_export("keras_core.ops.rfft")
def rfft(x, fft_length=None):
    """Real-valued Fast Fourier Transform along the last axis of the input.

    Computes the 1D Discrete Fourier Transform of a real-valued signal over the
    inner-most dimension of input.

    Since the Discrete Fourier Transform of a real-valued signal is
    Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique
    components of the FFT: the zero-frequency term, followed by the
    `fft_length / 2` positive-frequency terms.

    Along the axis RFFT is computed on, if `fft_length` is smaller than the
    corresponding dimension of the input, the dimension is cropped. If it is
    larger, the dimension is padded with zeros.

    Args:
        x: Input tensor.
        fft_length: An integer representing the number of the fft length. If not
            specified, it is inferred from the length of the last axis of `x`.
            Defaults to `None`.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Examples:

    >>> x = keras_core.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> rfft(x)
    (array([10.0, -2.5, -2.5]), array([0.0, 3.4409548, 0.81229924]))

    >>> rfft(x, 3)
    (array([3.0, -1.5]), array([0.0, 0.8660254]))
    """
    if any_symbolic_tensors((x,)):
        return RFFT(fft_length).symbolic_call(x)
    return backend.math.rfft(x, fft_length)


class STFT(Operation):
    def __init__(
        self,
        sequence_length,
        sequence_stride,
        fft_length,
        window="hann",
        center=True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input shape = {x.shape}"
            )
        if x.shape[-1] is not None:
            padded = 0 if self.center is False else (self.fft_length // 2) * 2
            num_sequences = (
                1
                + (x.shape[-1] + padded - self.fft_length)
                // self.sequence_stride
            )
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.fft_length // 2 + 1)
        return (
            KerasTensor(shape=new_shape, dtype=x.dtype),
            KerasTensor(shape=new_shape, dtype=x.dtype),
        )

    def call(self, x):
        return backend.math.stft(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            fft_length=self.fft_length,
            window=self.window,
            center=self.center,
        )


@keras_core_export("keras_core.ops.stft")
def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    """Short-Time Fourier Transform along the last axis of the input.

    The STFT computes the Fourier transform of short overlapping windows of the
    input. This giving frequency components of the signal as they change over
    time.

    Args:
        x: Input tensor.
        sequence_length: An integer representing the sequence length.
        sequence_stride: An integer representing the sequence hop size.
        fft_length: An integer representing the size of the FFT to apply. If not
            specified, uses the smallest power of 2 enclosing `sequence_length`.
        window: A string, a tensor of the window or `None`. If `window` is a
            string, available values are `"hann"` and `"hamming"`. If `window`
            is a tensor, it will be used directly as the window and its length
            must be `sequence_length`. If `window` is `None`, no windowing is
            used. Defaults to `"hann"`.
        center: Whether to pad `x` on both sides so that the t-th sequence is
            centered at time `t * sequence_stride`. Otherwise, the t-th sequence
            begins at time `t * sequence_stride`. Defaults to `True`.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        STFT output.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> stft(x, 3, 2, 3)
    (array([[0.75, -0.375],
       [3.75, -1.875],
       [5.25, -2.625]]), array([[0.0, 0.64951905],
       [0.0, 0.64951905],
       [0.0, -0.64951905]]))
    """
    if any_symbolic_tensors((x,)):
        return STFT(
            sequence_length, sequence_stride, fft_length, center, window
        ).symbolic_call(x)
    return backend.math.stft(
        x,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        fft_length=fft_length,
        window=window,
        center=center,
    )


class Rsqrt(Operation):
    """Computes reciprocal of square root of x element-wise.

    Args:
        x: input tensor

    Returns:
        A tensor with the same type as `x`.

    Example:

   >>> data = keras_core.ops.convert_to_tensor([1.0,10.0,100.0])
   >>> keras_core.ops.rsqrt(data)
   Array([1.        , 0.31622776, 0.1       ], dtype=float32)
   
    """

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.math.rsqrt(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export("keras_core.ops.rsqrt")
def rsqrt(x):
    if any_symbolic_tensors((x,)):
        return Rsqrt().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return backend.math.rsqrt(x)
