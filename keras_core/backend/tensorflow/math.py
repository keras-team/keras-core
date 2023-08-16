import tensorflow as tf

from keras_core.backend import standardize_dtype
from keras_core.backend.tensorflow.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        return tf.math.segment_sum(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        return tf.math.segment_max(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_max(data, segment_ids, num_segments)


def top_k(x, k, sorted=True):
    return tf.math.top_k(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    return tf.math.in_top_k(targets, predictions, k)


def logsumexp(x, axis=None, keepdims=False):
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    if mode == "reduced":
        return tf.linalg.qr(x)
    return tf.linalg.qr(x, full_matrices=True)


def extract_sequences(x, sequence_length, sequence_stride):
    return tf.signal.frame(
        x,
        frame_length=sequence_length,
        frame_step=sequence_stride,
        axis=-1,
        pad_end=False,
    )


def overlap_sequences(x, sequence_stride):
    return tf.signal.overlap_and_add(x, frame_step=sequence_stride)


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not real.dtype.is_floating or not imag.dtype.is_floating:
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = tf.dtypes.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft2d(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def rfft(x, fft_length=None):
    if fft_length is not None:
        fft_length = [fft_length]
    complex_output = tf.signal.rfft(x, fft_length=fft_length)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    if fft_length is not None:
        fft_length = [fft_length]
    return tf.signal.irfft(complex_input, fft_length)


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    if standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )
    x = convert_to_tensor(x)

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = tf.pad(x, pad_width, mode="reflect")

    x = extract_sequences(x, fft_length, sequence_stride)

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
            else:
                win = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
        else:
            win = convert_to_tensor(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        win = tf.pad(win, [[l_pad, r_pad]])
        x = tf.multiply(x, win)

    return rfft(x, fft_length)


def rsqrt(x):
    return tf.math.rsqrt(x)
