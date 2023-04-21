import tensorflow as tf

from keras_core import backend


def remove_squeezable_dimensions(
    labels, predictions, expected_rank_diff=0, name=None
):
    """Squeeze last dim if ranks differ from expected by exactly 1.

    In the common case where we expect shapes to match, `expected_rank_diff`
    defaults to 0, and we squeeze the last dimension of the larger rank if they
    differ by 1.

    But, for example, if `labels` contains class IDs and `predictions` contains
    1 probability per class, we expect `predictions` to have 1 more dimension
    than `labels`, so `expected_rank_diff` would be 1. In this case, we'd
    squeeze `labels` if `rank(predictions) - rank(labels) == 0`, and
    `predictions` if `rank(predictions) - rank(labels) == 2`.

    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.

    Args:
      labels: Label values, a `Tensor` whose dimensions match `predictions`.
      predictions: Predicted values, a `Tensor` of arbitrary dimensions.
      expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.
      name: Name of the op.

    Returns:
      Tuple of `labels` and `predictions`, possibly with last dim squeezed.
    """
    with backend.name_scope(name or "remove_squeezable_dimensions"):
        predictions_shape = predictions.shape
        predictions_rank = predictions_shape.ndims
        labels_shape = labels.shape
        labels_rank = labels_shape.ndims
        if (labels_rank is not None) and (predictions_rank is not None):
            # Use static rank.
            rank_diff = predictions_rank - labels_rank
            if rank_diff == expected_rank_diff + 1 and predictions_shape.dims[
                -1
            ].is_compatible_with(1):
                predictions = tf.squeeze(predictions, [-1])
            elif rank_diff == expected_rank_diff - 1 and labels_shape.dims[
                -1
            ].is_compatible_with(1):
                labels = tf.squeeze(labels, [-1])
            return labels, predictions

        # Use dynamic rank.
        rank_diff = tf.rank(predictions) - tf.rank(labels)
        if (predictions_rank is None) or (
            predictions_shape.dims[-1].is_compatible_with(1)
        ):
            predictions = tf.cond(
                tf.equal(expected_rank_diff + 1, rank_diff),
                lambda: tf.squeeze(predictions, [-1]),
                lambda: predictions,
            )
        if (labels_rank is None) or (
            labels_shape.dims[-1].is_compatible_with(1)
        ):
            labels = tf.cond(
                tf.equal(expected_rank_diff - 1, rank_diff),
                lambda: tf.squeeze(labels, [-1]),
                lambda: labels,
            )
        return labels, predictions


def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight=None):
    """Squeeze or expand last dimension if needed.

    1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1
    (using `remove_squeezable_dimensions`).
    2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
    from the new rank of `y_pred`.
    If `sample_weight` is scalar, it is kept scalar.

    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.

    Args:
      y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
      y_true: Optional label `Tensor` whose dimensions match `y_pred`.
      sample_weight: Optional weight scalar or `Tensor` whose dimensions match
        `y_pred`.

    Returns:
      Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
      the last dimension squeezed,
      `sample_weight` could be extended by one dimension.
      If `sample_weight` is None, (y_pred, y_true) is returned.
    """
    y_pred_shape = y_pred.shape
    y_pred_rank = y_pred_shape.ndims
    if y_true is not None:
        # If sparse matrix is provided as `y_true`, the last dimension in
        # `y_pred` may be > 1. Eg: y_true = [0, 1, 2] (shape=(3,)), y_pred =
        # [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]] (shape=(3, 3)) In
        # this case, we should not try to remove squeezable dimension.
        y_true_shape = y_true.shape
        y_true_rank = y_true_shape.ndims
        if (y_true_rank is not None) and (y_pred_rank is not None):
            # Use static rank for `y_true` and `y_pred`.
            if (y_pred_rank - y_true_rank != 1) or y_pred_shape[-1] == 1:
                y_true, y_pred = remove_squeezable_dimensions(y_true, y_pred)
        else:
            # Use dynamic rank.
            rank_diff = tf.rank(y_pred) - tf.rank(y_true)

            def squeeze_dims():
                return remove_squeezable_dimensions(y_true, y_pred)

            is_last_dim_1 = tf.equal(1, tf.shape(y_pred)[-1])

            def maybe_squeeze_dims():
                return tf.cond(
                    is_last_dim_1, squeeze_dims, lambda: (y_true, y_pred)
                )

            y_true, y_pred = tf.cond(
                tf.equal(1, rank_diff), maybe_squeeze_dims, squeeze_dims
            )

    if sample_weight is None:
        return y_pred, y_true

    weights_shape = sample_weight.shape
    weights_rank = weights_shape.ndims
    if weights_rank == 0:  # If weights is scalar, do nothing.
        return y_pred, y_true, sample_weight

    if (y_pred_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - y_pred_rank == 1:
            sample_weight = tf.squeeze(sample_weight, [-1])
        elif y_pred_rank - weights_rank == 1:
            sample_weight = tf.expand_dims(sample_weight, [-1])
        return y_pred, y_true, sample_weight

    # Use dynamic rank.
    weights_rank_tensor = tf.rank(sample_weight)
    rank_diff = weights_rank_tensor - tf.rank(y_pred)

    def maybe_squeeze_weights():
        return tf.squeeze(sample_weight, [-1])

    def _maybe_expand_weights():
        def expand_weights():
            return tf.expand_dims(sample_weight, [-1])

        return tf.cond(
            tf.equal(rank_diff, -1), expand_weights, lambda: sample_weight
        )

    def _maybe_adjust_weights():
        return tf.cond(
            tf.equal(rank_diff, 1), maybe_squeeze_weights, _maybe_expand_weights
        )

    # squeeze or expand last dim of `sample_weight` if its rank differs by 1
    # from the new rank of `y_pred`.
    sample_weight = tf.cond(
        tf.equal(weights_rank_tensor, 0),
        lambda: sample_weight,
        _maybe_adjust_weights,
    )
    return y_pred, y_true, sample_weight
