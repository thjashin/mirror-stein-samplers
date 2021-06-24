import tensorflow as tf

from .kernel import rbf


@tf.function
def svgd_update(target, eta, theta, kernel=rbf):
    K = theta.shape[-2]
    # grad_logp: [..., K, D]
    grad_logp_eta = target.dual_grad_logp(eta, theta=theta)
    # gram: [..., K, K], grad_gram: [..., K, K, D]
    gram, grad_gram, kernel_width2 = kernel(theta, theta, left_grad=True)
    # nabla2_psi_inv: [..., K, D, D]
    nabla2_psi_theta_inv = target.mirror_map.nabla2_psi_inv(theta)
    # repulsive_term: [..., K, D]
    repulsive_term = tf.einsum("...iab,...ijb->...ja", nabla2_psi_theta_inv, grad_gram) / K
    # weighted_grad: [..., K, D]
    weighted_grad = tf.matmul(gram, grad_logp_eta) / K
#     print(weighted_grad.dtype, repulsive_term.dtype)
#     tf.print("grad_term:", weighted_grad)
#     tf.print("repul_term:", repulsive_term)
    return weighted_grad + repulsive_term


@tf.function
def proj_svgd_update(target, theta, kernel=rbf):
    K = theta.shape[-2]
    # grad_logp: [..., K, D]
    grad_logp_theta = target.grad_logp(theta)
    # gram: [..., K, K], grad_gram: [..., K, K, D]
    gram, grad_gram, kernel_width2 = kernel(theta, theta, left_grad=True)
    # repulsive_term: [..., K, D]
    repulsive_term = tf.reduce_mean(grad_gram, axis=-3)
    # weighted_grad: [..., K, D]
    weighted_grad = tf.matmul(gram, grad_logp_theta) / K
#     print(weighted_grad.dtype, repulsive_term.dtype)
#     tf.print("grad_term:", weighted_grad)
#     tf.print("repul_term:", repulsive_term)
    return weighted_grad + repulsive_term
