import tensorflow as tf

from constrained.sampling.kernel import compute_squared_dist, cross_diff


@tf.function
def energy_dist(x, y):
    # x: [M, D]
    # y: [N, D]
    # m = x.shape[0]
    # n = y.shape[0]
    with tf.device('/CPU:0'):
    #     xx = tf.reduce_sum(tf.sqrt(compute_squared_dist(x, x))) / (m * (m - 1))
        xx = tf.reduce_mean(tf.sqrt(compute_squared_dist(cross_diff(x, x))))
    #     yy = tf.reduce_sum(tf.sqrt(compute_squared_dist(y, y))) / (n * (n - 1))
        yy = tf.reduce_mean(tf.sqrt(compute_squared_dist(cross_diff(y, y))))
        xy = tf.reduce_mean(tf.sqrt(compute_squared_dist(cross_diff(x, y))))
        return 2 * xy - xx - yy
