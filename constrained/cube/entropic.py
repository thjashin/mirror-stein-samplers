import tensorflow as tf


def safe_log(x):
    return tf.math.log(tf.maximum(tf.constant(1e-128, dtype=x.dtype), x))


def safe_reciprocal(x):
    return 1. / tf.maximum(x, tf.constant(1e-128, dtype=x.dtype))


class UnitCubeEntropic:
    def psi_star(self, eta):
        # ret: [..., K]
        # eta: [..., K, D]
        raise NotImplementedError()

    def nabla_psi(self, theta):
        # ret: [..., K, D]
        return safe_log(1. + theta) - safe_log(1. - theta)

    def nabla_psi_star(self, eta):
        # ret: [..., K, D]
        return 1. - 2. / (tf.exp(eta) + 1.)

    def nabla2_psi(self, theta):
        # theta: [..., K, D]
        # ret: [..., K, D, D]
        return tf.linalg.diag(2. * safe_reciprocal(1. - theta**2))

    def nabla2_psi_inv(self, theta):
        # theta: [..., K, D]
        # ret: [..., K, D, D]
        return tf.linalg.diag((1. - theta**2) * 0.5)

    def nabla2_psi_inv_mul(self, theta, rhs):
        # rhs: [..., K, D, ?]
        return ((1. - theta**2) * 0.5)[..., None] * rhs

    def logdet_nabla2_psi_star(self, eta):
        # eta: [..., K, D]
        # ret: [..., K]
        return tf.reduce_sum(tf.math.log(2.) + eta - 2 * tf.nn.softplus(eta), axis=-1)

    def grad_logdet_nabla2_psi_star(self, eta, theta=None):
        # eta: [..., K, D]
        # ret: [..., K, D]
        return -1. * theta

    def grad_nabla2_psi_inv_diag(self, theta):
        # theta: [..., K, D]
        return -tf.linalg.diag(theta)
