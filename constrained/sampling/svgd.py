import tensorflow as tf

from .kernel import imq


@tf.function
def svgd_update(target, eta, theta, kernel=imq, kernel_width2=None):
    # Standard MSVGD
    K = theta.shape[-2]
    # grad_logp: [..., K, D]
    grad_logp_eta = target.dual_grad_logp(eta, theta=theta)
    # gram: [..., K, K], grad_gram: [..., K, K, D]
    gram, grad_gram, *_ = kernel(theta, theta, left_grad=True, kernel_width2=kernel_width2)
    # nabla2_psi_inv: [..., K, D, D]
    nabla2_psi_theta_inv = target.mirror_map.nabla2_psi_inv(theta)
    # repulsive_term: [..., K, D]
    repulsive_term = tf.einsum("...iab,...ijb->...ja", nabla2_psi_theta_inv, grad_gram) / K
    # weighted_grad: [..., K, D]
    weighted_grad = tf.matmul(gram, grad_logp_eta) / K
    return weighted_grad + repulsive_term


def msvgd_eta_update(target, eta, theta, eta_kernel=imq, kernel_width2=None):
    # MSVGD with kernel k2 (kernel directly applied to eta)
    K = theta.shape[-2]
    # grad_logp: [..., K, D]
    grad_logp_eta = target.dual_grad_logp(eta, theta=theta)
    # gram: [..., K, K], grad_gram: [..., K, K, D]
    gram, grad_gram, *_ = eta_kernel(eta, eta, left_grad=True, kernel_width2=kernel_width2)
    # repulsive_term: [..., K, D]
    repulsive_term = tf.reduce_mean(grad_gram, axis=-3)
    # weighted_grad: [..., K, D]
    weighted_grad = tf.matmul(gram, grad_logp_eta) / K
    return weighted_grad + repulsive_term


@tf.function
def proj_svgd_update(target, theta, kernel=imq, kernel_width2=None):
    K = theta.shape[-2]
    # grad_logp: [..., K, D]
    grad_logp_theta = target.grad_logp(theta)
    # gram: [..., K, K], grad_gram: [..., K, K, D]
    gram, grad_gram, *_ = kernel(theta, theta, left_grad=True, kernel_width2=kernel_width2)
    # repulsive_term: [..., K, D]
    repulsive_term = tf.reduce_mean(grad_gram, axis=-3)
    # weighted_grad: [..., K, D]
    weighted_grad = tf.matmul(gram, grad_logp_theta) / K
    return weighted_grad + repulsive_term
