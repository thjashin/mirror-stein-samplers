import tensorflow as tf

from constrained.target import Target


class SelectiveTarget(Target):
    def __init__(self, nonneg_map, A, b, noise_scale):
        self.A = tf.constant(A, dtype=tf.float64)
        self.b = tf.constant(b, dtype=tf.float64)
        self.noise_scale = tf.constant(noise_scale, dtype=tf.float64)
        super(SelectiveTarget, self).__init__(nonneg_map)

    def logp(self, theta):
        # theta: [..., K, D]
        # ret: [..., K]
        recon = tf.einsum("...ki,ji->...kj", theta, self.A) + self.b
        return -0.5 * tf.reduce_sum(tf.square(recon), axis=-1) / tf.square(self.noise_scale)

    def grad_logp(self, theta):
        # ret: [..., K, D]
        # recon = theta @ tf.transpose(self.A) + self.b
        recon = tf.einsum("...ki,ji->...kj", theta, self.A) + self.b
        # ret = -recon @ self.A / tf.square(self.noise_scale)
        ret = -tf.einsum("...kj,ji->...ki", recon, self.A) / tf.square(self.noise_scale)
#         tf.print(ret)
#         tf.print("check:", super(QuadraticTarget, self).nabla_psi_inv_grad_logp(theta))
        return ret
