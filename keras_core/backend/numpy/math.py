import numpy as np


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


def _get_complex_array_from_tuple(a):
    if not isinstance(a, (tuple, list)) or len(a) != 2:
        raise ValueError(
            "Input `a` should be a tuple of two tensors - real and imaginary."
            f"Received: a={a}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = a

    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `a` should be a tuple of two arrays - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: a[0].shape = {real.shape}, a[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not np.issubdtype(real.dtype, np.floating) or not np.issubdtype(
        imag.dtype, np.floating
    ):
        raise ValueError(
            "At least one array in input `a` is not of type float."
            f"Received: a={a}."
        )
    complex_input = real + 1j * imag
    return complex_input


def fft(a):
    complex_input = _get_complex_array_from_tuple(a)
    complex_output = np.fft.fft(complex_input)
    return np.real(complex_output), np.imag(complex_output)


def fft2(a):
    complex_input = _get_complex_array_from_tuple(a)
    complex_output = np.fft.fft2(complex_input)
    return np.real(complex_output), np.imag(complex_output)
