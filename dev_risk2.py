import tensorflow_ranking as tfr
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow_ranking.python.losses_impl import neural_sort
from metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import norm
import math
def getGeoRisk(mat, alpha):
    ##### IMPORTANT
    # This function takes a matrix of number of rows as a number of queries, and the number of collumns as the number of systems.
    ##############
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0] * numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i] = np.sum(mat[:, i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = np.sum(mat[j, :])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd = norm.cdf(zRisk[i] / c)
        geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

    return geoRisk

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


cast_infer1_np = np.random.uniform(low=-1, high=1, size=(25, 10))
cast_infer2_np = np.random.uniform(low=-1, high=1, size=(25, 10))
cast_infer3_np = np.random.uniform(low=-1, high=1, size=(25, 10))
rate_batch_np = np.random.randint(5, size=(25, 10))

config = tf.ConfigProto()

# # cosine_similarity(result.T / result.max(), [ndcg_arr])
with tf.Session(config=config) as sess:
    rate_batch = tf.dtypes.cast(rate_batch_np, tf.float32)
    cast_infer1 = tf.dtypes.cast(cast_infer1_np, tf.float32)
    cast_infer2 = tf.dtypes.cast(cast_infer2_np, tf.float32)
    cast_infer3 = tf.dtypes.cast(cast_infer3_np, tf.float32)

    c1, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer1, [25, 10]))
    c2, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer2, [25, 10]))
    c3, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer3, [25, 10]))

    cc, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(rate_batch, [25, 10]))

    c1 = c1/tf.reduce_max(c1)
    c2 = c2/tf.reduce_max(c2)
    c3 = c1/tf.reduce_max(c3)

    cc = cc/tf.reduce_max(cc)

    # mat = tf.squeeze(tf.stack([c1, c2, c3]))
    mat = tf.squeeze(tf.stack([c1, c2, c3, cc]))
    mat = tf.transpose(mat)
    # result = sess.run(losses)
    #########

    def zRisk(mat, alpha, i=0):
        # alpha_tensor = torch.tensor([alpha], requires_grad=requires_grad, dtype=torch.float, device=device)
        alpha_tensor = tf.dtypes.cast(alpha, tf.float32)
        # si = torch.sum(mat[:, i])
        si = tf.reduce_sum(mat[:, i])

        # tj = torch.sum(mat, dim=1)
        tj = tf.reduce_sum(mat, axis=1)

        # n = torch.sum(tj)
        n = tf.reduce_sum(tj)

        xij_eij = mat[:, i] - si * (tj / n)
        subden = si * (tj / n)
        # den = torch.sqrt(subden + 1e-10)
        den = tf.math.sqrt(subden + 1e-10)
        # u = (den == 0) * torch.tensor([9e10], dtype=torch.float, requires_grad=requires_grad, device=device)
        # u = (den == 0) * tf.dtypes.cast(9e10, tf.float32)
        u = tf.dtypes.cast((den == 0), tf.float32) * tf.dtypes.cast(9e10, tf.float32)

        den = u + den
        div = xij_eij / den

        # less0 = (mat[:, i] - si * (tj / n)) / (den) < 0
        less0 = (mat[:, i] - si * (tj / n)) / (den) < 0
        # less0 = alpha_tensor * less0
        less0 = alpha_tensor * tf.dtypes.cast(less0, tf.float32)

        z_risk = div * less0 + div
        # z_risk = torch.sum(z_risk)
        z_risk = tf.reduce_sum(z_risk)

        return z_risk


    def geoRisk(mat, alpha, i=0):
        # mat = mat * (mat > 0)
        mat = mat * tf.dtypes.cast((mat > 0), tf.float32)
        # si = torch.sum(mat[:, i])
        si = tf.reduce_sum(mat[:, i])
        z_risk = zRisk(mat, alpha, i=i)

        num_queries = mat.shape[0]
        value = z_risk / num_queries
        # m = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        m = tf.distributions.Normal(0.0, 1.0)
        ncd = m.cdf(value)
        # return torch.sqrt((si / num_queries) * ncd + DEFAULT_EPS)
        return tf.math.sqrt((si / num_queries) * ncd + 1e-10)

    u_0 = geoRisk(mat, 5)
    u_last = geoRisk(mat, 5, i=-1)
    #########

    k = 10
    ndcg_arr = np.asarray([ndcg_score(x, y, k=k) for x, y in zip(rate_batch_np.tolist(), cast_infer1_np.tolist())])
    ndcg_mean = np.mean(ndcg_arr)
    x = 0
