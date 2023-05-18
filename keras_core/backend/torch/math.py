import torch

from keras_core.backend.torch.core import scatter, convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    num_repeats = torch.prod(torch.tensor(data.shape[1:]))
    segment_ids = segment_ids.repeat_interleave(num_repeats).view(
        segment_ids.shape[0], *data.shape[1:]
    )
    shape = [num_segments] + list(data.shape[1:])
    return scatter(data, segment_ids, shape=shape)


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    return torch.topk(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets)
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

    max_x = torch.max(x, dim=axis, keepdim=True)
    result = (
        torch.log(torch.sum(torch.exp(x - max_x), dim=axis, keepdim=keepdims))
        + max_x
    )
    return torch.squeeze(result, dim=axis) if not keepdims else result
