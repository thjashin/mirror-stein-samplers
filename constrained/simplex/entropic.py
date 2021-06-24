import tensorflow as tf


def safe_log(x):
    return tf.math.log(tf.maximum(tf.constant(1e-32, dtype=x.dtype), x))
#     return tf.math.log(x)


def safe_reciprocal(x):
    return 1. / tf.maximum(x, tf.constant(1e-32, dtype=x.dtype))
#     return 1. / x


class SimplexEntropicMap:
    def psi_star(self, eta):
        # ret: [..., K]
        # eta_ext: [..., K, D]
        eta_ext = tf.concat([eta, tf.zeros(tf.concat([eta.shape[:-1], [1]], -1), dtype=tf.float64)], -1)
        return tf.reduce_logsumexp(eta_ext, axis=-1)

    def nabla_psi(self, theta):
        # ret: [..., K, D - 1]
        return safe_log(theta) - safe_log(1. - tf.reduce_sum(theta, axis=-1, keepdims=True))

    def nabla_psi_star(self, eta):
        # ret: [..., K, D - 1]
        eta_ext = tf.concat([eta, tf.zeros(tf.concat([eta.shape[:-1], [1]], -1), dtype=tf.float64)], -1)
        return tf.nn.softmax(eta_ext)[..., :, :eta.shape[-1]]

    def nabla2_psi(self, theta):
        # theta: [..., K, D - 1]
        # ret: [..., K, D - 1, D - 1]
        return tf.linalg.diag(safe_reciprocal(theta)) + safe_reciprocal(
            1. - tf.reduce_sum(theta, axis=-1))[..., None, None]

    def nabla2_psi_inv(self, theta):
        # theta: [..., K, D - 1]
        # ret: [..., K, D - 1, D - 1]
        return tf.linalg.diag(theta) - theta[..., :, None] * theta[..., None, :]

    def logdet_nabla2_psi_star(self, eta):
        # eta: [..., K, D - 1]
        # ret: [..., K]
        return tf.reduce_sum(eta, axis=-1) - (eta.shape[-1] + 1) * self.psi_star(eta)

    def grad_logdet_nabla2_psi_star(self, eta, theta=None):
        # eta: [..., K, D - 1]
        # ret: [..., K, D - 1]
        D = eta.shape[-1] + 1
        if theta is None:
            theta = self.nabla_psi_star(eta)
        return 1. - D * theta

    def grad_nabla2_psi_inv_diag(self, theta):
        # theta: [..., K, D - 1]
        return -theta[..., None] - tf.linalg.diag(theta) + tf.eye(theta.shape[-1], dtype=tf.float64)
        # grad_nabla2_psi_inv_check = tf.linalg.diag_part(tape.batch_jacobian(nabla2_psi_theta_inv, theta))
