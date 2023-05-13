import jax
from jax import lax
import tensorflow as tf
from jax import numpy as jnp

def _jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]

def _with_rank_at_least(input, rank):
    """Returns a shape based on `self` with at least the given rank.

    Args:
        input: a jax array.
        rank: An integer.

    Returns:
        A shape that is at least as specific as `self` with at least the given
        rank.

    Raises:
        ValueError: If `self` does not represent a shape with at least the given
        `rank`.
    """
    if len(input.shape) is not None and len(input.shape) < rank:
        raise ValueError("Shape %s must have rank at least %d" % (input.shape, rank))
    else:
        return input
def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return jnp.transpose(input_t, axes)

    if not time_major:
        inputs = tf.nest.map_structure(swap_batch_timestep, inputs)

    flatted_inputs = tf.nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = flatted_inputs[0].shape[0]

    for input_ in flatted_inputs:
        _with_rank_at_least(input_, 3)

    if mask is not None:
        if mask.dtype != jnp.dtype('bool'):
            mask = mask.astype('bool')
        if len(mask.shape) == 2:
            mask = mask[..., None]
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    # tf.where needs its condition tensor to be the same shape as its two
    # result tensors, but in our case the condition (mask) tensor is
    # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
    # So we need to broadcast the mask to match the shape of inputs.
    # That's what the tile call does, it just repeats the mask along its
    # second dimension n times.
    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if tf.nest.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tf.nest.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = mask_t[..., None]
        multiples = [1] * fixed_dim + input_t.shape[fixed_dim:]
        return jnp.tile(mask_t, multiples)

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
            input_t = _jax_unstack(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
            return input_t

        if tf.nest.is_nested(inputs):
            processed_input = tf.nest.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tf.nest.pack_sequence_as(inputs, inp)

        if mask is not None:
            mask_list = _jax_unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = jnp.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = jnp.where(tiled_mask_t, output, prev_output)

                flat_states = tf.nest.flatten(states)
                flat_new_states = tf.nest.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    jnp.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )
                )
                states = tf.nest.pack_sequence_as(states, flat_final_states)

                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = jnp.stack(successive_outputs)

            if zero_output_for_mask:
                last_output = jnp.where(
                    _expand_mask(mask_list[-1], last_output),
                    last_output,
                    jnp.zeros_like(last_output),
                )
                outputs = jnp.where(
                    _expand_mask(mask, outputs, fixed_dim=2),
                    outputs,
                    jnp.zeros_like(outputs),
                )

        else:  # mask is None
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = jnp.stack(successive_outputs)

    else:  # Unroll == False
        states = tuple(initial_states)

        # Create input tensor array, if the inputs is nested tensors, then it
        # will be flattened first, and tensor array will be created one per
        # flattened tensor.
        input_ta = tuple(
            _jax_unstack(input_)
            if not go_backwards
            else _jax_unstack(jnp.flip(input_, axis=0))
            for input_ in flatted_inputs
        )

        # Get the time(0) input and compute the output for that, the output will
        # be used to determine the dtype of output tensor array. Don't read from
        # input_ta due to TensorArray clear_after_read default to True.
        input_time_zero = tf.nest.pack_sequence_as(
            inputs, [inp[0] for inp in flatted_inputs]
        )
        # output_time_zero is used to determine the cell output shape and its
        # dtype.  the value is discarded.
        output_time_zero, _ = step_function(
            input_time_zero, tuple(initial_states) + tuple(constants)
        )

        output_ta_size = time_steps_t if return_all_outputs else 1
        output_ta = tuple(
            [[jnp.zeros(out.shape)] * time_steps_t]
            for i, out in enumerate(tf.nest.flatten(output_time_zero))
        )

        time = 0

        if input_length is None:
            max_iterations = time_steps_t
        else:
            max_iterations = jnp.max(input_length)

        if mask is not None:
            if go_backwards:
                mask = jnp.flip(mask, axis=0)

            mask_ta = _jax_unstack(mask)

            def masking_fn(time):
                return mask_ta[time]

            def compute_masked_output(mask_t, flat_out, flat_mask):
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
                    for o in flat_out
                )
                return tuple(
                    jnp.where(m, o, fm)
                    for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask)
                )

        elif isinstance(input_length, jnp.ndarray):
            if go_backwards:
                max_len = input_length.max(axis=0)
                rev_input_length = (max_len - 1) - input_length
                def masking_fn(time):
                    return rev_input_length < time
            else:
                def masking_fn(time):
                    return input_length > time

            def compute_masked_output(mask_t, flat_out, flat_mask):
                return tuple(
                    jnp.where(mask_t, o, zo)
                    for (o, zo) in zip(flat_out, flat_mask)
                )

        else:
            masking_fn = None

        if masking_fn is not None:
            # Mask for the T output will be base on the output of T - 1. In the
            # case T = 0, a zero filled tensor will be used.
            flat_zero_output = tuple(
                jnp.zeros_like(o) for o in tf.nest.flatten(output_time_zero)
            )

            def _step(time, output_ta_t, prev_output, *states):
                current_input = tuple(ta[time] for ta in input_ta)
                # maybe set shape.
                current_input = tf.nest.pack_sequence_as(inputs, current_input)
                mask_t = masking_fn(time)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                # mask output
                flat_output = tf.nest.flatten(output)
                flat_mask_output = (
                    flat_zero_output
                    if zero_output_for_mask
                    else tf.nest.flatten(prev_output)
                )
                flat_new_output = compute_masked_output(
                    mask_t, flat_output, flat_mask_output
                )

                # mask states
                flat_state = tf.nest.flatten(states)
                flat_new_state = tf.nest.flatten(new_states)
                flat_final_state = compute_masked_output(
                    mask_t, flat_new_state, flat_state
                )
                new_states = tf.nest.pack_sequence_as(
                    new_states, flat_final_state
                )

                ta_index_to_write = time if return_all_outputs else 0
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_new_output)
                )

                return (time + 1, output_ta_t, tuple(flat_new_output)) + tuple(
                    new_states
                )

            final_outputs = lax.while_loop(
                body_fun=_step,
                init_val=(time, output_ta, flat_zero_output) + states,
                cond_fun=lambda args: args[0] < time_steps_t
            )
            # Skip final_outputs[2] which is the output for final timestep.
            new_states = final_outputs[3:]
        else:
            def _step(inputs):
                time, output_ta_t, *state = inputs[0], inputs[1], inputs[2:]
                current_input = tuple(ta[time] for ta in input_ta)

                current_input = tf.nest.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_new_state = tf.nest.flatten(new_states)

                flat_output = tf.nest.flatten(output)
                ta_index_to_write = time if return_all_outputs else 0
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_output)
                )

                new_states = tf.nest.pack_sequence_as(
                    initial_states, flat_new_state
                )
                return (time + 1, output_ta_t) + tuple(new_states)
            final_outputs = lax.while_loop(
                body_fun=_step,
                init_val=(time, output_ta) + states,
                cond_fun=lambda args: args[0] < time_steps_t
            )
            new_states = final_outputs[2:]

        output_ta = final_outputs[1]

        outputs = tuple(jnp.stack(o) for o in output_ta)
        last_output = tuple(o[-1] for o in outputs)

        outputs = tf.nest.pack_sequence_as(output_time_zero, outputs)
        last_output = tf.nest.pack_sequence_as(output_time_zero, last_output)

    if not time_major:
        outputs = tf.nest.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states
