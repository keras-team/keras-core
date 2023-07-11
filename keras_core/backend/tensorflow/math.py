import tensorflow as tf


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        return tf.math.segment_sum(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def top_k(x, k, sorted=True):
    return tf.math.top_k(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    return tf.math.in_top_k(targets, predictions, k)


def logsumexp(x, axis=None, keepdims=False):
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    if mode == "reduced":
        return tf.linalg.qr(x)
    return tf.linalg.qr(x, full_matrices=True)


def fft(a, n=None, axis=-1, norm=None):
    if n is not None:
        raise ValueError(
            "`n` argument value not supported. "
            f"Expected `None`. Received: n={n}"
        )
    if axis != -1:
        raise ValueError(
            "`axis` argument value not supported. "
            f"Expected `-1`. Received: axis={axis}"
        )
    if norm is not None:
        raise ValueError(
            "`norm` argument value not supported. "
            f"Expected `None`. Received: norm={norm}"
        )

    return tf.signal.fft(a)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    if s is not None:
        raise ValueError(
            "`s` argument value not supported. "
            f"Expected `None`. Received: s={s}"
        )
    if axes != (-2, -1):
        raise ValueError(
            "`axes` argument value not supported. "
            f"Expected `-1`. Received: axes={axes}"
        )
    if norm is not None:
        raise ValueError(
            "`norm` argument value not supported. "
            f"Expected `None`. Received: norm={norm}"
        )
    return tf.signal.fft2d(a)


def fftn(a, s=None, axes=None, norm=None):
    if s is not None:
        raise ValueError(
            "`s` argument value not supported. "
            f"Expected `None`. Received: s={s}"
        )

    if axes is None:
        axes = tuple(range(-len(a.shape), 0))
    if axes not in [(-1,), (-2, -1), (-3, -2, -1)]:
        raise ValueError(
            "`axes` argument value not supported. "
            f"`axes` should be one of `(-1, )`, `(-2, -1)`, `(-3, -2, -1)`. "
            f"Received: axes={axes}"
        )
    if norm is not None:
        raise ValueError(
            "`norm` argument value not supported. "
            f"Expected `None`. Received: norm={norm}"
        )

    if len(axes) == 1:
        return tf.signal.fft(a)
    elif len(axes) == 2:
        return tf.signal.fft2d(a)
    return tf.signal.fft3d(a)


def ifft(a, n=None, axis=-1, norm=None):
    if n is not None:
        raise ValueError(
            "`n` argument value not supported. "
            f"Expected `None`. Received: n={n}"
        )
    if axis != -1:
        raise ValueError(
            "`axis` argument value not supported. "
            f"Expected `-1`. Received: axis={axis}"
        )
    if norm is not None:
        raise ValueError(
            "`norm` argument value not supported. "
            f"Expected `None`. Received: norm={norm}"
        )

    return tf.signal.ifft(a)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    if s is not None:
        raise ValueError(
            "`s` argument value not supported. "
            f"Expected `None`. Received: s={s}"
        )
    if axes != (-2, -1):
        raise ValueError(
            "`axes` argument value not supported. "
            f"Expected `-1`. Received: axes={axes}"
        )
    if norm is not None:
        raise ValueError(
            "`norm` argument value not supported. "
            f"Expected `None`. Received: norm={norm}"
        )
    return tf.signal.ifft2d(a)


def ifftn(a, s=None, axes=None, norm=None):
    if s is not None:
        raise ValueError(
            "`s` argument value not supported. "
            f"Expected `None`. Received: s={s}"
        )

    if axes is None:
        axes = tuple(range(-len(a.shape), 0))
    if axes not in [(-1,), (-2, -1), (-3, -2, -1)]:
        raise ValueError(
            "`axes` argument value not supported. "
            f"`axes` should be one of `(-1, )`, `(-2, -1)`, `(-3, -2, -1)`. "
            f"Received: axes={axes}"
        )
    if norm is not None:
        raise ValueError(
            "`norm` argument value not supported. "
            f"Expected `None`. Received: norm={norm}"
        )

    if len(axes) == 1:
        return tf.signal.ifft(a)
    elif len(axes) == 2:
        return tf.signal.ifft2d(a)
    return tf.signal.ifft3d(a)
