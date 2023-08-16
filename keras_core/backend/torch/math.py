import math

import torch

from keras_core.backend import standardize_dtype
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import get_device
from keras_core.backend.torch.numpy import pad


def segment_sum(data, segment_ids, num_segments=None, **kwargs):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    num_repeats = torch.prod(
        torch.tensor(data.shape[1:], device=get_device())
    ).long()
    # To use `scatter_add` in torch, we need to replicate `segment_ids` into the
    # shape of `data`.
    segment_ids = (
        segment_ids.repeat_interleave(num_repeats)
        .view(*data.shape)
        .type(torch.int64)
    )
    num_segments = num_segments or len(torch.unique(segment_ids))

    # .scatter_add does not support -1 in the indices.
    # Add all out-of-bound indices value to an extra dimension after
    # num_segments, which is removed before returning the result.

    # Replacing the out-of-bound indices.
    segment_ids = torch.where(segment_ids >= 0, segment_ids, num_segments)
    segment_ids = torch.where(
        segment_ids < num_segments, segment_ids, num_segments
    )

    # Add one more dimension to the result shape with the "+1".
    shape = (num_segments + 1,) + tuple(data.shape[1:])

    result = torch.zeros(*shape, device=get_device()).scatter_add(
        0, segment_ids, data.float()
    )

    # Removing the extra dimension.
    result = result[:-1, ...]

    return result.type(data.dtype)


def segment_max(data, segment_ids, num_segments=None, **kwargs):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    num_repeats = torch.prod(
        torch.tensor(data.shape[1:], device=get_device())
    ).long()
    # To use `scatter_reduce` in torch, we need to replicate `segment_ids` into
    # the shape of `data`.
    segment_ids = (
        segment_ids.repeat_interleave(num_repeats)
        .view(*data.shape)
        .type(torch.int64)
    )
    num_segments = num_segments or len(torch.unique(segment_ids))

    # .scatter_reduce does not support -1 in the indices.
    # Add all out-of-bound indices value to an extra dimension after
    # num_segments, which is removed before returning the result.

    # Replacing the out-of-bound indices.
    segment_ids = torch.where(segment_ids >= 0, segment_ids, num_segments)
    segment_ids = torch.where(
        segment_ids < num_segments, segment_ids, num_segments
    )

    # Add one more dimension to the result shape with the "+1".
    shape = (num_segments + 1,) + tuple(data.shape[1:])

    result = torch.zeros(*shape, device=get_device()).scatter_reduce(
        0, segment_ids, data.float(), "amax"
    )

    # Removing the extra dimension.
    result = result[:-1, ...]

    return result.type(data.dtype)


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    return torch.topk(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets).type(torch.int64)
    targets = targets[:, None]
    predictions = convert_to_tensor(predictions)
    topk_values = top_k(predictions, k).values
    targets_values = torch.take_along_dim(predictions, targets, dim=-1)
    mask = targets_values >= topk_values
    return torch.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is None:
        max_x = torch.max(x)
        return torch.log(torch.sum(torch.exp(x - max_x))) + max_x

    max_x = torch.amax(x, dim=axis, keepdim=True)
    result = (
        torch.log(torch.sum(torch.exp(x - max_x), dim=axis, keepdim=True))
        + max_x
    )
    return torch.squeeze(result, dim=axis) if not keepdims else result


def qr(x, mode="reduced"):
    x = convert_to_tensor(x)
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    x = convert_to_tensor(x)
    return torch.linalg.qr(x, mode=mode)


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    return torch.unfold_copy(
        x, dimension=-1, size=sequence_length, step=sequence_stride
    )


def overlap_sequences(x, sequence_stride):
    # Ref: https://github.com/google/jax/blob/main/jax/_src/scipy/signal.py
    x = convert_to_tensor(x)
    *batch_shape, nframes, segment_len = x.size()
    flat_batchsize = math.prod(batch_shape)
    x = torch.reshape(x, (flat_batchsize, nframes, segment_len))
    output_size = sequence_stride * (nframes - 1) + segment_len
    nstep_per_segment = 1 + (segment_len - 1) // sequence_stride
    # Here, we use shorter notation for axes.
    # B: batch_size, N: nframes, S: nstep_per_segment,
    # T: segment_len divided by S
    padded_segment_len = nstep_per_segment * sequence_stride
    x = torch.nn.functional.pad(
        x, (0, padded_segment_len - segment_len, 0, 0, 0, 0)
    )
    x = torch.reshape(
        x, (flat_batchsize, nframes, nstep_per_segment, sequence_stride)
    )
    # For obtaining shifted signals, this routine reinterprets flattened array
    # with a shrinked axis.  With appropriate truncation/ padding, this
    # operation pushes the last padded elements of the previous row to the head
    # of the current row.
    # See implementation of `overlap_and_add` in Tensorflow for details.
    x = torch.permute(x, (0, 2, 1, 3))  # x: (B, S, N, T)
    x = torch.nn.functional.pad(x, (0, 0, 0, nframes, 0, 0, 0, 0))
    # x: (B, S, N*2, T)
    shrinked = x.shape[2] - 1
    x = torch.reshape(x, (flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = torch.reshape(
        x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
    )
    # Finally, sum shifted segments, and truncate results to the output_size.
    x = torch.sum(x, dim=1)[:, :output_size]
    return torch.reshape(x, tuple(batch_shape) + (-1,))


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
    # Check shape.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not torch.is_floating_point(real) or not torch.is_floating_point(imag):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )

    complex_input = torch.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = torch.fft.fft(complex_input)
    return complex_output.real, complex_output.imag


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = torch.fft.fft2(complex_input)
    return complex_output.real, complex_output.imag


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = torch.fft.rfft(x, n=fft_length, dim=-1, norm="backward")
    return complex_output.real, complex_output.imag


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    return torch.fft.irfft(complex_input, n=fft_length, dim=-1, norm="backward")


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
        pad_width = [(0, 0) for _ in range(x.ndim)]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        # torch does not support reflect padding when x.ndim >= 3
        if x.ndim < 3:
            x = pad(x, pad_width, "reflect")
        else:
            x = pad(x, pad_width, "constant")

    x = extract_sequences(x, fft_length, sequence_stride)

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win = torch.hann_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
                )
            else:
                win = torch.hamming_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
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
        win = pad(win, [[l_pad, r_pad]], "constant")
        x = torch.multiply(x, win)

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
            if window == "hann":
                win = torch.hann_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
                )
            else:
                win = torch.hamming_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
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
        win = pad(win, [[l_pad, r_pad]], "constant")

        # square and sum
        _sequence_length = sequence_length + l_pad + r_pad
        denom = torch.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = pad(denom, [(0, overlaps * sequence_stride - _sequence_length)])
        denom = torch.reshape(denom, [overlaps, sequence_stride])
        denom = torch.sum(denom, 0, keepdims=True)
        denom = torch.tile(denom, [overlaps, 1])
        denom = torch.reshape(denom, [overlaps * sequence_stride])
        win = torch.divide(win, denom[:_sequence_length])
        x = torch.multiply(x, win)

    x = overlap_sequences(x, sequence_stride)

    if center:
        x[..., fft_length // 2 : -(fft_length // 2)]
    return x


def rsqrt(x):
    x = convert_to_tensor(x)
    return torch.rsqrt(x)
