import math

import jax
import jax.numpy as jnp

from keras_core.backend import standardize_dtype
from keras_core.backend.jax.core import convert_to_tensor
from keras_core.utils.module_utils import scipy


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        raise ValueError(
            "Argument `num_segments` must be set when using the JAX backend. "
            "Received: num_segments=None"
        )
    return jax.ops.segment_sum(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        raise ValueError(
            "Argument `num_segments` must be set when using the JAX backend. "
            "Received: num_segments=None"
        )
    return jax.ops.segment_max(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def top_k(x, k, sorted=True):
    # Jax does not supported `sorted`, but in the case where `sorted=False`,
    # order is not guaranteed, so OK to return sorted output.
    return jax.lax.top_k(x, k)


def in_top_k(targets, predictions, k):
    targets = targets[..., None]
    topk_values = top_k(predictions, k)[0]
    targets_values = jnp.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return jnp.any(mask, axis=1)


def logsumexp(x, axis=None, keepdims=False):
    max_x = jnp.max(x, axis=axis, keepdims=True)
    result = (
        jnp.log(jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=True)) + max_x
    )
    return jnp.squeeze(result) if not keepdims else result


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return jnp.linalg.qr(x, mode=mode)


def extract_sequences(x, sequence_length, sequence_stride):
    *batch_shape, signal_length = x.shape
    batch_shape = list(batch_shape)
    x = jnp.reshape(x, (math.prod(batch_shape), signal_length, 1))
    x = jax.lax.conv_general_dilated_patches(
        x,
        (sequence_length,),
        (sequence_stride,),
        "VALID",
        dimension_numbers=("NTC", "OIT", "NTC"),
    )
    return jnp.reshape(x, (*batch_shape, *x.shape[-2:]))


def overlap_sequences(x, sequence_stride):
    # Ref: https://github.com/google/jax/blob/main/jax/_src/scipy/signal.py
    *batch_shape, nframes, segment_len = x.shape
    flat_batchsize = math.prod(batch_shape)
    x = jnp.reshape(x, (flat_batchsize, nframes, segment_len))
    output_size = sequence_stride * (nframes - 1) + segment_len
    nstep_per_segment = 1 + (segment_len - 1) // sequence_stride
    # Here, we use shorter notation for axes.
    # B: batch_size, N: nframes, S: nstep_per_segment,
    # T: segment_len divided by S
    padded_segment_len = nstep_per_segment * sequence_stride
    x = jnp.pad(x, ((0, 0), (0, 0), (0, padded_segment_len - segment_len)))
    x = jnp.reshape(
        x, (flat_batchsize, nframes, nstep_per_segment, sequence_stride)
    )
    # For obtaining shifted signals, this routine reinterprets flattened array
    # with a shrinked axis.  With appropriate truncation/ padding, this
    # operation pushes the last padded elements of the previous row to the head
    # of the current row.
    # See implementation of `overlap_and_add` in Tensorflow for details.
    x = jnp.transpose(x, (0, 2, 1, 3))  # x: (B, S, N, T)
    x = jnp.pad(x, ((0, 0), (0, 0), (0, nframes), (0, 0)))
    # x: (B, S, N*2, T)
    shrinked = x.shape[2] - 1
    x = jnp.reshape(x, (flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = jnp.reshape(
        x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
    )
    # Finally, sum shifted segments, and truncate results to the output_size.
    x = jnp.sum(x, axis=1)[:, :output_size]
    return jnp.reshape(x, tuple(batch_shape) + (-1,))


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
    if not jnp.issubdtype(real.dtype, jnp.floating) or not jnp.issubdtype(
        imag.dtype, jnp.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = jax.lax.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = jnp.fft.fft(complex_input)
    return jnp.real(complex_output), jnp.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = jnp.fft.fft2(complex_input)
    return jnp.real(complex_output), jnp.imag(complex_output)


def rfft(x, fft_length=None):
    complex_output = jnp.fft.rfft(x, n=fft_length, axis=-1, norm="backward")
    return jnp.real(complex_output), jnp.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    return jnp.fft.irfft(complex_input, n=fft_length, axis=-1, norm="backward")


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
        x = jnp.pad(x, pad_width, mode="reflect")

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
        win = jnp.pad(win, [[l_pad, r_pad]])
        x = jnp.multiply(x, win)

    return rfft(x, fft_length)


def istft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    # ref:
    # torch: aten/src/ATen/native/SpectralOps.cpp
    # tf: tf.signal.inverse_stft_window_fn
    x = irfft(x, fft_length)

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
        win = jnp.pad(win, [[l_pad, r_pad]])

        # square and sum
        _sequence_length = sequence_length + l_pad + r_pad
        denom = jnp.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = jnp.pad(
            denom, [(0, overlaps * sequence_stride - _sequence_length)]
        )
        denom = jnp.reshape(denom, [overlaps, sequence_stride])
        denom = jnp.sum(denom, 0, keepdims=True)
        denom = jnp.tile(denom, [overlaps, 1])
        denom = jnp.reshape(denom, [overlaps * sequence_stride])
        win = jnp.divide(win, denom[:_sequence_length])
        x = jnp.multiply(x, win)

    x = overlap_sequences(x, sequence_stride)

    if center:
        x[..., fft_length // 2 : -(fft_length // 2)]
    return x


def rsqrt(x):
    return jax.lax.rsqrt(x)
