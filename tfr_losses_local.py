import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

from tensorflow_ranking.python.losses_impl import neural_sort, _pairwise_comparison, _compute_ranks, \
    gumbel_softmax_sample
from tensorflow_ranking.python.losses import create_ndcg_lambda_weight
from tensorflow_ranking.python.utils import is_label_valid, approx_ranks, ndcg

_EPSILON = 1e-10


def GumbelApproxNDCGLossLocal(labels, logits):
    labels, logits, gbl_weights = gumbel_softmax_sample(
        labels, logits, None)

    alpha = 10.0
    is_valid = is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    ranks = approx_ranks(logits, alpha=alpha)

    return -ndcg(labels, ranks), tf.reshape(
        tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])


def PairwiseLogisticLossLocal(labels, logits, lambda_weight=True):
    def _pairwise_loss(pairwise_logits):
        """See `_PairwiseLoss`."""
        # The following is the same as log(1 + exp(-pairwise_logits)).
        return tf.nn.relu(-pairwise_logits) + tf.math.log1p(
            tf.exp(-tf.abs(pairwise_logits)))

    is_valid = is_label_valid(labels)
    ranks = _compute_ranks(logits, is_valid)
    pairwise_labels, pairwise_logits = _pairwise_comparison(labels, logits)
    pairwise_weights = pairwise_labels

    _lambda_weight = None
    if lambda_weight:
        _lambda_weight = create_ndcg_lambda_weight()

    if _lambda_weight is not None:
        pairwise_weights *= _lambda_weight.pair_weights(labels, ranks)
        pairwise_weights *= tf.cast(tf.shape(input=labels)[1], dtype=tf.float32)

    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return _pairwise_loss(pairwise_logits), pairwise_weights


def NeuralSortCrossEntropyLossLocal(labels, logits, temperature=1.0):
    temperature = temperature
    is_valid = is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(is_valid, labels, -1e3 * tf.ones_like(labels))

    # shape = [batch_size, list_size, list_size].
    true_perm = neural_sort(labels, temperature=temperature)
    smooth_perm = neural_sort(logits, temperature=temperature)
    losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=true_perm, logits=tf.math.log(1e-20 + smooth_perm), axis=2)
    # shape = [batch_size, list_size].
    losses = tf.reduce_mean(input_tensor=losses, axis=-1, keepdims=True)

    return losses, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])
