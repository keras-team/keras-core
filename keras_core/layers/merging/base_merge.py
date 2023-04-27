from keras_core import backend
from keras_core import operations as ops
from keras_core.layers.layer import Layer


class _Merge(Layer):
    """Generic merge layer for elementwise merge functions.
    Used to implement `Sum`, `Average`, etc.
    """

    def __init__(self, **kwargs):
        """Initializes a Merge layer.
        Args:
          **kwargs: standard layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.
        Args:
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor
        Returns:
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.
        Raises:
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[: -len(shape2)])
        for i, j in zip(shape1[-len(shape2) :], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError(
                        "Inputs have incompatible shapes. "
                        f"Received shapes {shape1} and {shape2}"
                    )
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape[0], tuple):
            raise ValueError(
                "A merge layer should be called on a list of inputs. "
                f"Received: input_shape={input_shape} (not a list of shapes)"
            )
        if len(input_shape) < 1:
            raise ValueError(
                "A merge layer should be called "
                "on a list of at least 1 input. "
                f"Got {len(input_shape)} inputs. "
                f"Full input_shape received: {input_shape}"
            )

        batch_sizes = {s[0] for s in input_shape if s} - {None}
        if len(batch_sizes) > 1:
            raise ValueError(
                "Cannot merge tensors with different batch sizes. "
                f"Got tensors with shapes {input_shape}"
            )

        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]

        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(
                output_shape, shape
            )

        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "A merge layer should be called on a list of inputs. "
                f"Received: inputs={inputs} (not a list of tensors)"
            )
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(ops.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = ops.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = ops.expand_dims(x, axis=1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... ,
                # batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = ops.ndim(x)

                    if x_ndim is None:
                        x_shape = ops.shape(x)
                        batch_size = x_shape[0]

                        new_shape = backend.concatenate(
                            [x_shape[1:], ops.expand_dims(batch_size, axis=-1)]
                        )
                        x_transposed = ops.reshape(
                            x,
                            ops.stack(
                                [batch_size, ops.prod(x_shape[1:])],
                                axis=0,
                            ),
                        )
                        x_transposed = ops.transpose(x_transposed, perm=(1, 0))
                        x_transposed = ops.reshape(x_transposed, new_shape)

                        reshaped_inputs.append(x_transposed)
                        transposed = True

                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(ops.transpose(x, perm=dims))
                        print(dims)
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or
                        # scalars.
                        reshaped_inputs.append(x)

                y = self._merge_function(reshaped_inputs)
                y_ndim = ops.ndim(y)

                if transposed:
                    # If inputs have been transposed, we have to transpose the
                    # output too.
                    if y_ndim is None:
                        y_shape = ops.shape(y)
                        y_ndim = ops.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = ops.concatenate(
                            [
                                ops.expand_dims(batch_size, axis=-1),
                                y_shape[: y_ndim - 1],
                            ]
                        )
                        y = ops.reshape(y, (-1, batch_size))
                        y = ops.transpose(y, perm=(1, 0))
                        y = ops.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = ops.transpose(y, perm=dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]

        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(
                output_shape, shape
            )
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, (tuple, list)):
            raise ValueError(f"`mask` should be a list. Received: mask={mask}")
        if not isinstance(inputs, (tuple, list)):
            raise ValueError(
                f"`inputs` should be a list. Received: inputs={inputs}"
            )
        if len(mask) != len(inputs):
            raise ValueError(
                "The lists `inputs` and `mask` should have the same length. "
                f"Received: inputs={inputs} of length {len(inputs)}, and "
                f"mask={mask} of length {len(mask)}"
            )
        if all(m is None for m in mask):
            return None
        masks = [ops.expand_dims(m, axis=0) for m in mask if m is not None]
        return ops.all(ops.concatenate(masks, axis=0), axis=0, keepdims=False)

    def get_config(self):
        return super().get_config()
