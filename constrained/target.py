import tensorflow as tf


class Target:
    def __init__(self, mirror_map):
        self._mirror_map = mirror_map

    @property
    def mirror_map(self):
        return self._mirror_map

    def logp(self, theta):
        # theta: [..., K, D]
        # ret: [..., K]
        raise NotImplementedError()

    def grad_logp(self, theta):
        # ret: [..., K, D]
        with tf.GradientTape() as tape:
            tape.watch(theta)
            logp_theta = tf.reduce_sum(self.logp(theta))
        return tape.gradient(logp_theta, theta)

    def nabla_psi_inv_grad_logp(self, theta):
        # ret: [..., K, D]
        grad_logp_theta = self.grad_logp(theta)
        return tf.squeeze(self._mirror_map.nabla2_psi_inv_mul(theta, grad_logp_theta[..., None]), axis=-1)

    def dual_logp(self, eta, theta=None):
        # ret: [..., K]
        if theta is None:
            theta = self._mirror_map.nabla_psi_star(eta)
        return self.logp(theta) + self._mirror_map.logdet_nabla2_psi_star(eta)

    def dual_grad_logp(self, eta, theta=None):
        # ret: [..., K, D]
        if theta is None:
            theta = self._mirror_map.nabla_psi_star(eta)
        return self.nabla_psi_inv_grad_logp(theta) + self._mirror_map.grad_logdet_nabla2_psi_star(eta, theta=theta)
