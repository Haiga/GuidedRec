import tensorflow_ranking as tfr
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow_ranking.python.losses_impl import neural_sort


def NeuralSortCrossEntropyLossLocal(labels, logits, temperature=1.0):
    def is_label_valid(labels):
        """Returns a boolean `Tensor` for label validity."""
        labels = tf.convert_to_tensor(value=labels)
        return tf.greater_equal(labels, 0.)

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


cast_infer = np.random.uniform(low=-1, high=1, size=(25, 10))
rate_batch = np.random.randint(5, size=(25, 10))

config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    rate_batch = tf.dtypes.cast(rate_batch, tf.float32)
    cast_infer = tf.dtypes.cast(cast_infer, tf.float32)

    c, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer, [25, 10]))
    result = sess.run(c)
    x = 0
