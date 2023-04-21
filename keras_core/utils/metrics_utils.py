# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils related to keras metrics."""

import tensorflow as tf
import numpy as np
from keras_core.utils.generic_utils import to_list
from keras_core import backend

def assert_thresholds_range(thresholds):
    if thresholds is not None:
        invalid_thresholds = [
            t for t in thresholds if t is None or t < 0 or t > 1
        ]
        if invalid_thresholds:
            raise ValueError(
                "Threshold values must be in [0, 1]. "
                f"Received: {invalid_thresholds}"
            )

def parse_init_thresholds(thresholds, default_threshold=0.5):
    if thresholds is not None:
        assert_thresholds_range(to_list(thresholds))
    thresholds = to_list(
        default_threshold if thresholds is None else thresholds
    )
    return thresholds

class ConfusionMatrix(Enum):
    TRUE_POSITIVES = "tp"
    FALSE_POSITIVES = "fp"
    TRUE_NEGATIVES = "tn"
    FALSE_NEGATIVES = "fn"

def _update_confusion_matrix_variables_optimized(
    variables_to_update,
    y_true,
    y_pred,
    thresholds,
    multi_label=False,
    sample_weights=None,
    label_weights=None,
    thresholds_with_epsilon=False,
):
    """Update confusion matrix variables with memory efficient alternative.

    Note that the thresholds need to be evenly distributed within the list, eg,
    the diff between consecutive elements are the same.

    To compute TP/FP/TN/FN, we are measuring a binary classifier
      C(t) = (predictions >= t)
    at each threshold 't'. So we have
      TP(t) = sum( C(t) * true_labels )
      FP(t) = sum( C(t) * false_labels )

    But, computing C(t) requires computation for each t. To make it fast,
    observe that C(t) is a cumulative integral, and so if we have
      thresholds = [t_0, ..., t_{n-1}];  t_0 < ... < t_{n-1}
    where n = num_thresholds, and if we can compute the bucket function
      B(i) = Sum( (predictions == t), t_i <= t < t{i+1} )
    then we get
      C(t_i) = sum( B(j), j >= i )
    which is the reversed cumulative sum in tf.cumsum().

    We can compute B(i) efficiently by taking advantage of the fact that
    our thresholds are evenly distributed, in that
      width = 1.0 / (num_thresholds - 1)
      thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
    Given a prediction value p, we can map it to its bucket by
      bucket_index(p) = floor( p * (num_thresholds - 1) )
    so we can use tf.math.unsorted_segment_sum() to update the buckets in one
    pass.

    Consider following example:
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.5, 0.3, 0.9]
    thresholds = [0.0, 0.5, 1.0]
    num_buckets = 2   # [0.0, 1.0], (1.0, 2.0]
    bucket_index(y_pred) = tf.math.floor(y_pred * num_buckets)
                         = tf.math.floor([0.2, 1.0, 0.6, 1.8])
                         = [0, 0, 0, 1]
    # The meaning of this bucket is that if any of the label is true,
    # then 1 will be added to the corresponding bucket with the index.
    # Eg, if the label for 0.2 is true, then 1 will be added to bucket 0. If the
    # label for 1.8 is true, then 1 will be added to bucket 1.
    #
    # Note the second item "1.0" is floored to 0, since the value need to be
    # strictly larger than the bucket lower bound.
    # In the implementation, we use tf.math.ceil() - 1 to achieve this.
    tp_bucket_value = tf.math.unsorted_segment_sum(true_labels, bucket_indices,
                                                   num_segments=num_thresholds)
                    = [1, 1, 0]
    # For [1, 1, 0] here, it means there is 1 true value contributed by bucket
    # 0, and 1 value contributed by bucket 1. When we aggregate them to
    # together, the result become [a + b + c, b + c, c], since large thresholds
    # will always contribute to the value for smaller thresholds.
    true_positive = tf.math.cumsum(tp_bucket_value, reverse=True)
                  = [2, 1, 0]

    This implementation exhibits a run time and space complexity of O(T + N),
    where T is the number of thresholds and N is the size of predictions.
    Metrics that rely on standard implementation instead exhibit a complexity of
    O(T * N).

    Args:
      variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
        and corresponding variables to update as values.
      y_true: A floating point `Tensor` whose shape matches `y_pred`. Will be
        cast to `bool`.
      y_pred: A floating point `Tensor` of arbitrary shape and whose values are
        in the range `[0, 1]`.
      thresholds: A sorted floating point `Tensor` with value in `[0, 1]`.
        It need to be evenly distributed (the diff between each element need to
        be the same).
      multi_label: Optional boolean indicating whether multidimensional
        prediction/labels should be treated as multilabel responses, or
        flattened into a single label. When True, the valus of
        `variables_to_update` must have a second dimension equal to the number
        of labels in y_true and y_pred, and those tensors must not be
        RaggedTensors.
      sample_weights: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions
        must be either `1`, or the same as the corresponding `y_true`
        dimension).
      label_weights: Optional tensor of non-negative weights for multilabel
        data. The weights are applied when calculating TP, FP, FN, and TN
        without explicit multilabel handling (i.e. when the data is to be
        flattened).
      thresholds_with_epsilon: Optional boolean indicating whether the leading
        and tailing thresholds has any epsilon added for floating point
        imprecisions.  It will change how we handle the leading and tailing
        bucket.

    Returns:
      Update op.
    """
    num_thresholds = thresholds.shape.as_list()[0]

    if sample_weights is None:
        sample_weights = 1.0
    else:
        sample_weights = tf.__internal__.ops.broadcast_weights(
            tf.cast(sample_weights, dtype=y_pred.dtype), y_pred
        )
        if not multi_label:
            sample_weights = tf.reshape(sample_weights, [-1])
    if label_weights is None:
        label_weights = 1.0
    else:
        label_weights = tf.expand_dims(label_weights, 0)
        label_weights = tf.__internal__.ops.broadcast_weights(
            label_weights, y_pred
        )
        if not multi_label:
            label_weights = tf.reshape(label_weights, [-1])
    weights = tf.cast(tf.multiply(sample_weights, label_weights), y_true.dtype)

    # We shouldn't need this, but in case there are predict value that is out of
    # the range of [0.0, 1.0]
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0.0, clip_value_max=1.0)

    y_true = tf.cast(tf.cast(y_true, tf.bool), y_true.dtype)
    if not multi_label:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

    true_labels = tf.multiply(y_true, weights)
    false_labels = tf.multiply((1.0 - y_true), weights)

    # Compute the bucket indices for each prediction value.
    # Since the predict value has to be strictly greater than the thresholds,
    # eg, buckets like [0, 0.5], (0.5, 1], and 0.5 belongs to first bucket.
    # We have to use math.ceil(val) - 1 for the bucket.
    bucket_indices = tf.math.ceil(y_pred * (num_thresholds - 1)) - 1

    if thresholds_with_epsilon:
        # In this case, the first bucket should actually take into account since
        # the any prediction between [0.0, 1.0] should be larger than the first
        # threshold. We change the bucket value from -1 to 0.
        bucket_indices = tf.nn.relu(bucket_indices)

    bucket_indices = tf.cast(bucket_indices, tf.int32)

    if multi_label:
        # We need to run bucket segment sum for each of the label class. In the
        # multi_label case, the rank of the label is 2. We first transpose it so
        # that the label dim becomes the first and we can parallel run though
        # them.
        true_labels = tf.transpose(true_labels)
        false_labels = tf.transpose(false_labels)
        bucket_indices = tf.transpose(bucket_indices)

        def gather_bucket(label_and_bucket_index):
            label, bucket_index = (
                label_and_bucket_index[0],
                label_and_bucket_index[1],
            )
            return tf.math.unsorted_segment_sum(
                data=label,
                segment_ids=bucket_index,
                num_segments=num_thresholds,
            )

        tp_bucket_v = tf.vectorized_map(
            gather_bucket, (true_labels, bucket_indices), warn=False
        )
        fp_bucket_v = tf.vectorized_map(
            gather_bucket, (false_labels, bucket_indices), warn=False
        )
        tp = tf.transpose(tf.cumsum(tp_bucket_v, reverse=True, axis=1))
        fp = tf.transpose(tf.cumsum(fp_bucket_v, reverse=True, axis=1))
    else:
        tp_bucket_v = tf.math.unsorted_segment_sum(
            data=true_labels,
            segment_ids=bucket_indices,
            num_segments=num_thresholds,
        )
        fp_bucket_v = tf.math.unsorted_segment_sum(
            data=false_labels,
            segment_ids=bucket_indices,
            num_segments=num_thresholds,
        )
        tp = tf.cumsum(tp_bucket_v, reverse=True)
        fp = tf.cumsum(fp_bucket_v, reverse=True)

    # fn = sum(true_labels) - tp
    # tn = sum(false_labels) - fp
    if (
        ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
        or ConfusionMatrix.FALSE_NEGATIVES in variables_to_update
    ):
        if multi_label:
            total_true_labels = tf.reduce_sum(true_labels, axis=1)
            total_false_labels = tf.reduce_sum(false_labels, axis=1)
        else:
            total_true_labels = tf.reduce_sum(true_labels)
            total_false_labels = tf.reduce_sum(false_labels)

    update_ops = []
    if ConfusionMatrix.TRUE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_POSITIVES]
        update_ops.append(variable.assign_add(tp))
    if ConfusionMatrix.FALSE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_POSITIVES]
        update_ops.append(variable.assign_add(fp))
    if ConfusionMatrix.TRUE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_NEGATIVES]
        tn = total_false_labels - fp
        update_ops.append(variable.assign_add(tn))
    if ConfusionMatrix.FALSE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_NEGATIVES]
        fn = total_true_labels - tp
        update_ops.append(variable.assign_add(fn))
    return tf.group(update_ops)


def is_evenly_distributed_thresholds(thresholds):
    """Check if the thresholds list is evenly distributed.

    We could leverage evenly distributed thresholds to use less memory when
    calculate metrcis like AUC where each individual threshold need to be
    evaluated.

    Args:
      thresholds: A python list or tuple, or 1D numpy array whose value is
        ranged in [0, 1].

    Returns:
      boolean, whether the values in the inputs are evenly distributed.
    """
    # Check the list value and see if it is evenly distributed.
    num_thresholds = len(thresholds)
    if num_thresholds < 3:
        return False
    even_thresholds = np.arange(num_thresholds, dtype=np.float32) / (
        num_thresholds - 1
    )
    return np.allclose(thresholds, even_thresholds, atol=backend.epsilon())

def update_confusion_matrix_variables(
    variables_to_update,
    y_true,
    y_pred,
    thresholds,
    top_k=None,
    class_id=None,
    sample_weight=None,
    multi_label=False,
    label_weights=None,
    thresholds_distributed_evenly=False,
):
    """Returns op to update the given confusion matrix variables.

    For every pair of values in y_true and y_pred:

    true_positive: y_true == True and y_pred > thresholds
    false_negatives: y_true == True and y_pred <= thresholds
    true_negatives: y_true == False and y_pred <= thresholds
    false_positive: y_true == False and y_pred > thresholds

    The results will be weighted and added together. When multiple thresholds
    are provided, we will repeat the same for every threshold.

    For estimation of these metrics over a stream of data, the function creates
    an `update_op` operation that updates the given variables.

    If `sample_weight` is `None`, weights default to 1.
    Use weights of 0 to mask values.

    Args:
      variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
        and corresponding variables to update as values.
      y_true: A `Tensor` whose shape matches `y_pred`. Will be cast to `bool`.
      y_pred: A floating point `Tensor` of arbitrary shape and whose values are
        in the range `[0, 1]`.
      thresholds: A float value, float tensor, python list, or tuple of float
        thresholds in `[0, 1]`, or NEG_INF (used when top_k is set).
      top_k: Optional int, indicates that the positive labels should be limited
        to the top k predictions.
      class_id: Optional int, limits the prediction and labels to the class
        specified by this argument.
      sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions
        must be either `1`, or the same as the corresponding `y_true`
        dimension).
      multi_label: Optional boolean indicating whether multidimensional
        prediction/labels should be treated as multilabel responses, or
        flattened into a single label. When True, the valus of
        `variables_to_update` must have a second dimension equal to the number
        of labels in y_true and y_pred, and those tensors must not be
        RaggedTensors.
      label_weights: (optional) tensor of non-negative weights for multilabel
        data. The weights are applied when calculating TP, FP, FN, and TN
        without explicit multilabel handling (i.e. when the data is to be
        flattened).
      thresholds_distributed_evenly: Boolean, whether the thresholds are evenly
        distributed within the list. An optimized method will be used if this is
        the case. See _update_confusion_matrix_variables_optimized() for more
        details.

    Returns:
      Update op.

    Raises:
      ValueError: If `y_pred` and `y_true` have mismatched shapes, or if
        `sample_weight` is not `None` and its shape doesn't match `y_pred`, or
        if `variables_to_update` contains invalid keys.
    """
    if multi_label and label_weights is not None:
        raise ValueError(
            "`label_weights` for multilabel data should be handled "
            "outside of `update_confusion_matrix_variables` when "
            "`multi_label` is True."
        )
    if variables_to_update is None:
        return
    if not any(
        key for key in variables_to_update if key in list(ConfusionMatrix)
    ):
        raise ValueError(
            "Please provide at least one valid confusion matrix "
            "variable to update. Valid variable key options are: "
            f'"{list(ConfusionMatrix)}". '
            f'Received: "{variables_to_update.keys()}"'
        )

    variable_dtype = list(variables_to_update.values())[0].dtype

    y_true = tf.cast(y_true, dtype=variable_dtype)
    y_pred = tf.cast(y_pred, dtype=variable_dtype)

    if thresholds_distributed_evenly:
        # Check whether the thresholds has any leading or tailing epsilon added
        # for floating point imprecision. The leading and tailing threshold will
        # be handled bit differently as the corner case.  At this point,
        # thresholds should be a list/array with more than 2 items, and ranged
        # between [0, 1]. See is_evenly_distributed_thresholds() for more
        # details.
        thresholds_with_epsilon = thresholds[0] < 0.0 or thresholds[-1] > 1.0

    thresholds = tf.convert_to_tensor(thresholds, dtype=variable_dtype)
    num_thresholds = thresholds.shape.as_list()[0]

    if multi_label:
        one_thresh = tf.equal(
            tf.cast(1, dtype=tf.int32),
            tf.rank(thresholds),
            name="one_set_of_thresholds_cond",
        )
    else:
        one_thresh = tf.cast(True, dtype=tf.bool)

    invalid_keys = [
        key for key in variables_to_update if key not in list(ConfusionMatrix)
    ]
    if invalid_keys:
        raise ValueError(
            f'Invalid keys: "{invalid_keys}". '
            f'Valid variable key options are: "{list(ConfusionMatrix)}"'
        )

    if sample_weight is None:
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )
    else:
        sample_weight = tf.cast(sample_weight, dtype=variable_dtype)
        (
            y_pred,
            y_true,
            sample_weight,
        ) = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true, sample_weight=sample_weight
        )
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    if top_k is not None:
        y_pred = _filter_top_k(y_pred, top_k)
    if class_id is not None:
        # Preserve dimension to match with sample_weight
        y_true = y_true[..., class_id, None]
        y_pred = y_pred[..., class_id, None]

    if thresholds_distributed_evenly:
        return _update_confusion_matrix_variables_optimized(
            variables_to_update,
            y_true,
            y_pred,
            thresholds,
            multi_label=multi_label,
            sample_weights=sample_weight,
            label_weights=label_weights,
            thresholds_with_epsilon=thresholds_with_epsilon,
        )

    pred_shape = tf.shape(y_pred)
    num_predictions = pred_shape[0]
    if y_pred.shape.ndims == 1:
        num_labels = 1
    else:
        num_labels = tf.math.reduce_prod(pred_shape[1:], axis=0)
    thresh_label_tile = tf.where(
        one_thresh, num_labels, tf.ones([], dtype=tf.int32)
    )

    # Reshape predictions and labels, adding a dim for thresholding.
    if multi_label:
        predictions_extra_dim = tf.expand_dims(y_pred, 0)
        labels_extra_dim = tf.expand_dims(tf.cast(y_true, dtype=tf.bool), 0)
    else:
        # Flatten predictions and labels when not multilabel.
        predictions_extra_dim = tf.reshape(y_pred, [1, -1])
        labels_extra_dim = tf.reshape(tf.cast(y_true, dtype=tf.bool), [1, -1])

    # Tile the thresholds for every prediction.
    if multi_label:
        thresh_pretile_shape = [num_thresholds, 1, -1]
        thresh_tiles = [1, num_predictions, thresh_label_tile]
        data_tiles = [num_thresholds, 1, 1]
    else:
        thresh_pretile_shape = [num_thresholds, -1]
        thresh_tiles = [1, num_predictions * num_labels]
        data_tiles = [num_thresholds, 1]

    thresh_tiled = tf.tile(
        tf.reshape(thresholds, thresh_pretile_shape), tf.stack(thresh_tiles)
    )

    # Tile the predictions for every threshold.
    preds_tiled = tf.tile(predictions_extra_dim, data_tiles)

    # Compare predictions and threshold.
    pred_is_pos = tf.greater(preds_tiled, thresh_tiled)

    # Tile labels by number of thresholds
    label_is_pos = tf.tile(labels_extra_dim, data_tiles)

    if sample_weight is not None:
        sample_weight = tf.__internal__.ops.broadcast_weights(
            tf.cast(sample_weight, dtype=variable_dtype), y_pred
        )
        weights_tiled = tf.tile(
            tf.reshape(sample_weight, thresh_tiles), data_tiles
        )
    else:
        weights_tiled = None

    if label_weights is not None and not multi_label:
        label_weights = tf.expand_dims(label_weights, 0)
        label_weights = tf.__internal__.ops.broadcast_weights(
            label_weights, y_pred
        )
        label_weights_tiled = tf.tile(
            tf.reshape(label_weights, thresh_tiles), data_tiles
        )
        if weights_tiled is None:
            weights_tiled = label_weights_tiled
        else:
            weights_tiled = tf.multiply(weights_tiled, label_weights_tiled)

    update_ops = []

    def weighted_assign_add(label, pred, weights, var):
        label_and_pred = tf.cast(tf.logical_and(label, pred), dtype=var.dtype)
        if weights is not None:
            label_and_pred *= tf.cast(weights, dtype=var.dtype)
        return var.assign_add(tf.reduce_sum(label_and_pred, 1))

    loop_vars = {
        ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos),
    }
    update_tn = ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
    update_fp = ConfusionMatrix.FALSE_POSITIVES in variables_to_update
    update_fn = ConfusionMatrix.FALSE_NEGATIVES in variables_to_update

    if update_fn or update_tn:
        pred_is_neg = tf.logical_not(pred_is_pos)
        loop_vars[ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)

    if update_fp or update_tn:
        label_is_neg = tf.logical_not(label_is_pos)
        loop_vars[ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
        if update_tn:
            loop_vars[ConfusionMatrix.TRUE_NEGATIVES] = (
                label_is_neg,
                pred_is_neg,
            )

    for matrix_cond, (label, pred) in loop_vars.items():

        if matrix_cond in variables_to_update:
            update_ops.append(
                weighted_assign_add(
                    label, pred, weights_tiled, variables_to_update[matrix_cond]
                )
            )

    return tf.group(update_ops)

def _filter_top_k(x, k):
    """Filters top-k values in the last dim of x and set the rest to NEG_INF.

    Used for computing top-k prediction values in dense labels (which has the
    same shape as predictions) for recall and precision top-k metrics.

    Args:
      x: tensor with any dimensions.
      k: the number of values to keep.

    Returns:
      tensor with same shape and dtype as x.
    """
    _, top_k_idx = tf.math.top_k(x, k, sorted=False)
    top_k_mask = tf.reduce_sum(
        tf.one_hot(top_k_idx, tf.shape(x)[-1], axis=-1), axis=-2
    )
    return x * top_k_mask + NEG_INF * (1 - top_k_mask)
