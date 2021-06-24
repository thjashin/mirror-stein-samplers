import tensorflow as tf


def safe_log(x):
    return tf.math.log(tf.maximum(tf.constant(1e-32, dtype=x.dtype), x))
#     return tf.math.log(x)


def safe_reciprocal(x):
    return 1. / tf.maximum(x, tf.constant(1e-32, dtype=x.dtype))
#     return 1. / x


class NonnegativeEntropicMap:
    def psi_star(self, eta):
        # ret: [..., K]
        return tf.reduce_sum(tf.exp(eta), axis=-1)

    def nabla_psi(self, theta):
        # ret: [..., K, D]
        return safe_log(theta)

    def nabla_psi_star(self, eta):
        # ret: [..., K, D]
        return tf.exp(eta)

    def nabla2_psi(self, theta):
        # theta: [..., K, D]
        # ret: [..., K, D, D]
        return tf.linalg.diag(safe_reciprocal(theta))

    def nabla2_psi_inv(self, theta):
        # theta: [..., K, D]
        # ret: [..., K, D, D]
        return tf.linalg.diag(theta)

    def nabla2_psi_inv_mul(self, theta, rhs):
        # rhs: [..., K, D, ?]
        return theta[..., None] * rhs

    def logdet_nabla2_psi_star(self, eta):
        # eta: [..., K, D]
        # ret: [..., K]
        return tf.reduce_sum(eta, axis=-1)

    def grad_logdet_nabla2_psi_star(self, eta, theta=None):
        # eta: [..., K, D]
        # ret: [..., K, D]
        return tf.ones_like(eta)

    def grad_nabla2_psi_inv_diag(self, theta):
        # ret: [..., K, D, D]
        return tf.eye(theta.shape[-1], batch_shape=theta.shape[:-1], dtype=tf.float64)
        # grad_nabla2_psi_inv_check = tf.linalg.diag_part(tape.batch_jacobian(nabla2_psi_theta_inv, theta))
