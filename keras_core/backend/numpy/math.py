import numpy as np


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        # Segment ids must be less than the number of segments.
        num_segments = np.amax(segment_ids) + 1

    if sorted:
        result = np.zeros(num_segments, dtype=data.dtype)
        np.add.at(result, segment_ids, data)
    else:
        sort_indices = np.argsort(segment_ids)
        sorted_segment_ids = segment_ids[sort_indices]
        sorted_data = data[sort_indices]

        result = np.zeros(num_segments, dtype=data.dtype)
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
    top_k_indices = top_k(predictions, k, sorted=True)[1]
    targets = targets[..., np.newaxis]
    mask = targets == top_k_indices
    return np.any(mask, axis=-1)
