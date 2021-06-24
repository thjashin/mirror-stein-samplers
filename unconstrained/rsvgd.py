import tensorflow as tf

from .kernel import rbf, heuristic_kernel_width


def rsvgd_update(theta, theta_grads, G, G_inv, grad_G_inv=None, kernel=rbf, grad_G_inv_trace=None, G_inv_grads=None):
    if kernel is not rbf:
        raise NotImplementedError()
    K = tf.cast(tf.shape(theta)[0], theta.dtype)
    # theta: [K, D]
    # theta_grads: [K, D]
    # G, G_inv: [K, D, D]
    # grad_G_inv: [K, D, D]
    # M*D, M*D*D, M*D, M*D
    gradp = theta_grads
    Ginv = G_inv
    if grad_G_inv_trace is not None:
        gradGinv = grad_G_inv_trace
    else:
        gradGinv = tf.reduce_sum(grad_G_inv, axis=-1)
    gradDet = -1. * tf.squeeze(G @ gradGinv[:, :, None], axis=-1)
    # https://github.com/changliu00/Riem-SVGD
    # totScore: M*D
    if gradp is None:
        totScore = G_inv_grads + tf.squeeze(0.5 * tf.matmul(gradDet[:, None, :], Ginv), axis=-2) + gradGinv
    else:
        totScore = tf.squeeze(tf.matmul((gradp + 0.5 * gradDet)[:, None, :], Ginv), axis=-2) + gradGinv
    # M*M*D, expectation over the first index
    Dxy = theta[:, None, :] - theta[None, :, :]
    # M*M
    sqDxy = tf.reduce_sum(Dxy**2, axis=-1)
    DxyGinv = tf.matmul(Dxy, Ginv)

    # h = np.median(sqDxy)
    # h = 0.5 * h / np.log(M + 1)
    h = 0.5 * heuristic_kernel_width(sqDxy)

    # compute the rbf kernel
    Kxy = tf.exp((-sqDxy / h) / 2)
    vect = tf.reduce_sum(tf.linalg.matrix_transpose(tf.reduce_sum((-DxyGinv / h + totScore[:, None, :]) * Dxy, axis=-1) * Kxy / h**2)[:, :, None] * DxyGinv, axis=1) \
        + tf.squeeze(tf.matmul(((Kxy / h) @ totScore)[:, None, :], Ginv), axis=-2) \
        + tf.reduce_sum(((tf.linalg.trace(Ginv) / h**2) * Kxy)[:, :, None] * DxyGinv, axis=-2) \
        - tf.squeeze(tf.matmul(tf.reduce_sum((2 / h**2) * Kxy[:, :, None] * DxyGinv, axis=-3)[:, None, :], Ginv), axis=-2)

    vect = vect / K
    return vect
