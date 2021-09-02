import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

def zRisk(mat, alpha, i=0):
    alpha_tensor = tf.dtypes.cast(alpha, tf.float32)
    si = tf.reduce_sum(mat[:, i])
    tj = tf.reduce_sum(mat, axis=1)
    n = tf.reduce_sum(tj)
    xij_eij = mat[:, i] - si * (tj / n)
    subden = si * (tj / n)
    den = tf.math.sqrt(subden + 1e-10)
    u = tf.dtypes.cast((den == 0), tf.float32) * tf.dtypes.cast(9e10, tf.float32)
    den = u + den
    div = xij_eij / den
    less0 = (mat[:, i] - si * (tj / n)) / (den) < 0
    less0 = alpha_tensor * tf.dtypes.cast(less0, tf.float32)
    z_risk = div * less0 + div
    z_risk = tf.reduce_sum(z_risk)
    return z_risk


def geoRisk(mat, alpha, i=0):
    mat = mat * tf.dtypes.cast((mat > 0), tf.float32)
    si = tf.reduce_sum(mat[:, i])
    z_risk = zRisk(mat, alpha, i=i)
    num_queries = tf.cast(mat.shape[0], tf.float32)
    value = z_risk / num_queries
    m = tf.distributions.Normal(0.0, 1.0)
    ncd = m.cdf(value)
    # return tf.math.sqrt((si / num_queries) * ncd + 1e-10)
    return (si / num_queries) * ncd
