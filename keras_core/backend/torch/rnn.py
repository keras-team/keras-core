import numpy as np
import torch

from keras_core.backend.torch.core import cast
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import to_torch_dtype


def rnn(*args, **kwargs):
    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(input_t.ndim))
        axes[0], axes[1] = 1, 0
        return torch.transpose(input_t, 0, 1)

    if not time_major:
        inputs = torch.vmap(swap_batch_timestep)(inputs)

    flattened_inputs = torch.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]

    if mask is not None:
        if mask.dtype != "bool":
            mask = mask.astype("bool")
        if len(mask.shape) == 2:
            mask = torch.unsqueeze(mask, -1)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if nest.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if nest.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = torch.unsqueeze(mask_t, -1)
        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        return torch.tile(mask_t, multiples)

    if unroll:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        # Process the input tensors. The input tensor need to be split on the
        # time_step dim, and reverse if go_backwards is True. In the case of
        # nested input, the input is flattened and then transformed
        # individually.  The result of this will be a tuple of lists, each of
        # the item in tuple is list of the tensor with shape (batch, feature)
        def _process_single_input_t(input_t):
            input_t = torch.unbind(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
            return input_t

        if nest.is_nested(inputs):
            processed_input = nest.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return nest.pack_sequence_as(inputs, inp)

def lstm(*args, **kwargs):
    raise NotImplementedError


def gru(*args, **kwargs):
    raise NotImplementedError
