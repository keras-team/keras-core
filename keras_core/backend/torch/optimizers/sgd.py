from typing import List
from typing import Optional

import torch
from torch import Tensor

from keras_core import optimizers


class SGD(optimizers.SGD):
    def apply(self, grads, trainable_variables=None):
        trainable_variables = [v.value for v in trainable_variables]
        apply_gradients(trainable_variables, grads)
        return self.iterations


def apply_gradients(params, grads):
    momentum_buffer_list = [None] * len(params)

    _multi_tensor_sgd(
        params,
        grads,
        momentum_buffer_list,
        weight_decay=0,
        momentum=0,
        lr=0.001,
        dampening=False,
        nesterov=False,
        maximize=False,
        has_sparse_grad=False,
    )


def _multi_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool
):
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(
                        grads[i]
                    ).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs don't support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)
