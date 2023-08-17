import math

import numpy as np

from keras_core.backend import standardize_dtype
from keras_core.backend.jax.math import fft as jax_fft
from keras_core.backend.jax.math import fft2 as jax_fft2
from keras_core.backend.numpy.core import convert_to_tensor
from keras_core.utils.module_utils import scipy


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        num_segments = np.amax(segment_ids) + 1

    valid_indices = segment_ids >= 0  # Ignore segment_ids that are -1
    valid_data = data[valid_indices]
    valid_segment_ids = segment_ids[valid_indices]

    data_shape = list(valid_data.shape)
    data_shape[
        0
    ] = num_segments  # Replace first dimension (which corresponds to segments)

    if sorted:
        result = np.zeros(data_shape, dtype=valid_data.dtype)
        np.add.at(result, valid_segment_ids, valid_data)
    else:
        sort_indices = np.argsort(valid_segment_ids)
        sorted_segment_ids = valid_segment_ids[sort_indices]
        sorted_data = valid_data[sort_indices]

        result = np.zeros(data_shape, dtype=valid_data.dtype)
        np.add.at(result, sorted_segment_ids, sorted_data)

    return result


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        num_segments = np.amax(segment_ids) + 1

    valid_indices = segment_ids >= 0  # Ignore segment_ids that are -1
    valid_data = data[valid_indices]
    valid_segment_ids = segment_ids[valid_indices]

    data_shape = list(valid_data.shape)
    data_shape[
        0
    ] = num_segments  # Replace first dimension (which corresponds to segments)

    if sorted:
        result = np.zeros(data_shape, dtype=valid_data.dtype)
        np.maximum.at(result, valid_segment_ids, valid_data)
    else:
        sort_indices = np.argsort(valid_segment_ids)
        sorted_segment_ids = valid_segment_ids[sort_indices]
        sorted_data = valid_data[sort_indices]

        result = np.zeros(data_shape, dtype=valid_data.dtype)
        np.maximum.at(result, sorted_segment_ids, sorted_data)

    return result


def top_k(x, k, sorted=False):
    sorted_indices = np.argsort(x, axis=-1)[..., ::-1]
    sorted_values = np.sort(x, axis=-1)[..., ::-1]

    if sorted:
        # Take the k largest values.
        top_k_values = sorted_values[..., :k]
        top_k_indices = sorted_indices[..., :k]
    else:
        # Partition the array such that all values larger than the k-th
        # largest value are to the right of it.
        top_k_values = np.partition(x, -k, axis=-1)[..., -k:]
        top_k_indices = np.argpartition(x, -k, axis=-1)[..., -k:]

        # Get the indices in sorted order.
        idx = np.argsort(-top_k_values, axis=-1)

        # Get the top k values and their indices.
        top_k_values = np.take_along_axis(top_k_values, idx, axis=-1)
        top_k_indices = np.take_along_axis(top_k_indices, idx, axis=-1)

    return top_k_values, top_k_indices


def in_top_k(targets, predictions, k):
    targets = targets[:, None]
    topk_values = top_k(predictions, k)[0]
    targets_values = np.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return np.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    max_x = np.max(x, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True)) + max_x
    return np.squeeze(result) if not keepdims else result


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return np.linalg.qr(x, mode=mode)


def extract_sequences(x, sequence_length, sequence_stride):
    *batch_shape, _ = x.shape
    batch_shape = list(batch_shape)
    shape = x.shape[:-1] + (
        (x.shape[-1] - (sequence_length - sequence_stride)) // sequence_stride,
        sequence_length,
    )
    strides = x.strides[:-1] + (
        sequence_stride * x.strides[-1],
        x.strides[-1],
    )
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.reshape(x, (*batch_shape, *x.shape[-2:]))


def overlap_sequences(x, sequence_stride):
    # Ref: https://github.com/google/jax/blob/main/jax/_src/scipy/signal.py
    *batch_shape, num_sequences, sequence_length = x.shape
    if sequence_stride > sequence_length:
        raise ValueError(
            "`sequence_stride` must not larger than x.shape[-1]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}"
        )
    if sequence_stride < (sequence_length / num_sequences):
        raise ValueError(
            "`sequence_stride` must not less than x.shape[-1] / x.shape[-2]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}, x.shape[-2]={num_sequences}"
        )
    flat_batchsize = math.prod(batch_shape)
    x = np.reshape(x, (flat_batchsize, num_sequences, sequence_length))
    output_size = sequence_stride * (num_sequences - 1) + sequence_length
    nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
    # Here, we use shorter notation for axes.
    # B: batch_size, N: nframes, S: nstep_per_segment,
    # T: segment_len divided by S
    padded_segment_len = nstep_per_segment * sequence_stride
    x = np.pad(x, ((0, 0), (0, 0), (0, padded_segment_len - sequence_length)))
    x = np.reshape(
        x, (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride)
    )
    # For obtaining shifted signals, this routine reinterprets flattened array
    # with a shrinked axis.  With appropriate truncation/ padding, this
    # operation pushes the last padded elements of the previous row to the head
    # of the current row.
    # See implementation of `overlap_and_add` in Tensorflow for details.
    x = x.transpose((0, 2, 1, 3))  # x: (B, S, N, T)
    x = np.pad(x, ((0, 0), (0, 0), (0, num_sequences), (0, 0)))
    # x: (B, S, N*2, T)
    shrinked = x.shape[2] - 1
    x = np.reshape(x, (flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = np.reshape(
        x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
    )
    # Finally, sum shifted segments, and truncate results to the output_size.
    x = np.sum(x, axis=1)[:, :output_size]
    return np.reshape(x, tuple(batch_shape) + (-1,))


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = x
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not np.issubdtype(real.dtype, np.floating) or not np.issubdtype(
        imag.dtype, np.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = real + 1j * imag
    return complex_input


def fft(x):
    real, imag = jax_fft(x)
    return np.array(real), np.array(imag)


def fft2(x):
    real, imag = jax_fft2(x)
    return np.array(real), np.array(imag)


def rfft(x, fft_length=None):
    complex_output = np.fft.rfft(x, n=fft_length, axis=-1, norm="backward")
    # numpy always outputs complex128, so we need to recast the dtype
    return (
        np.real(complex_output).astype(x.dtype),
        np.imag(complex_output).astype(x.dtype),
    )


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    # numpy always outputs float64, so we need to recast the dtype
    return np.fft.irfft(
        complex_input, n=fft_length, axis=-1, norm="backward"
    ).astype(x[0].dtype)


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

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = np.pad(x, pad_width, mode="reflect")

    x = extract_sequences(x, fft_length, sequence_stride)

    if window is not None:
        if isinstance(window, str):
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length), dtype=x.dtype
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
        win = np.pad(win, [[l_pad, r_pad]])
        x = np.multiply(x, win)

    return rfft(x, fft_length)


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    # ref:
    # torch: aten/src/ATen/native/SpectralOps.cpp
    # tf: tf.signal.inverse_stft_window_fn
    x = irfft(x, fft_length)

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)

    if window is not None:
        if isinstance(window, str):
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length), dtype=x.dtype
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
        win = np.pad(win, [[l_pad, r_pad]])

        # square and sum
        _sequence_length = sequence_length + l_pad + r_pad
        denom = np.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = np.pad(
            denom, [(0, overlaps * sequence_stride - _sequence_length)]
        )
        denom = np.reshape(denom, [overlaps, sequence_stride])
        denom = np.sum(denom, 0, keepdims=True)
        denom = np.tile(denom, [overlaps, 1])
        denom = np.reshape(denom, [overlaps * sequence_stride])
        win = np.divide(win, denom[:_sequence_length])
        x = np.multiply(x, win)

    x = overlap_sequences(x, sequence_stride)

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]
