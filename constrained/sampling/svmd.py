import tensorflow as tf
import numpy as np

from .kernel import nystrom, imq


@tf.function
def svmd_update(target, theta, kernel=imq, n_eigen_threshold=0.98, md=False, jitter=1e-5, kernel_width2=None, eigen_gpu=False):
    K = theta.shape[-2]
    # gram: [..., K, K]
    gram, grad_gram, *_ = kernel(theta, theta, left_grad=True, kernel_width2=kernel_width2)
    # tf.debugging.check_numerics(gram, "Gram matrix inf/nan")
    gram_jittered = gram + jitter * tf.eye(K, dtype=tf.float64)
    try:
        if eigen_gpu:
            eigval, eigvec = tf.linalg.eigh(gram_jittered)
        else:
            with tf.device("/cpu:0"):
                eigval, eigvec = tf.linalg.eigh(gram_jittered)
    except Exception as error:
        print(gram_jittered)
        raise error
    if n_eigen_threshold is not None:
        eigen_arr = tf.reduce_mean(tf.reshape(eigval, [-1, K]), axis=0)
        eigen_arr = tf.reverse(eigen_arr, axis=[-1])
        eigen_arr /= tf.reduce_sum(eigen_arr)
        eigen_cum = tf.cumsum(eigen_arr, axis=-1)
        n_eigen = tf.reduce_sum(
            tf.cast(tf.less(eigen_cum, n_eigen_threshold), tf.int32)) + 1
        # eigval: [..., n_eigen]
        # eigvec: [..., K, n_eigen]
        eigval = eigval[..., -n_eigen:]
        eigvec = eigvec[..., -n_eigen:]
    # mu: [..., n_eigen]
    mu = eigval / K
    # v: [..., K, n_eigen]
    v = eigvec * np.sqrt(K)
    # use nystrom formula to fix the gradient
    # v_theta: [..., K, n_eigen]
    # v_theta_grad: [..., K, n_eigen, D]
    _, v_theta_grad = nystrom(gram, eigval, eigvec, grad_Kxz=grad_gram)
    # nabla2_psi_theta: [..., K, D, D]
    nabla2_psi_theta = target.mirror_map.nabla2_psi(theta)
    mu_sqrt = tf.sqrt(mu)

    # nabla2_psi_theta_inv_grad: [..., K, D]
    nabla2_psi_theta_inv_grad = target.nabla_psi_inv_grad_logp(theta)
    # nabla2_psi_theta_inv: [..., K, D, D]
    nabla2_psi_theta_inv = target.mirror_map.nabla2_psi_inv(theta)
    # grad_nabla2_psi_inv: [..., K, D, D]
    grad_nabla2_psi_theta_inv_diag = target.mirror_map.grad_nabla2_psi_inv_diag(theta)

    # [..., K, K]
    i_reduced = tf.einsum("...i,...ki,...mi->...km", mu_sqrt, v, v)
    # weighted_grad: [..., K, D]
    weighted_grad = 1 / (K * K) * tf.einsum(
        "...km,...j,...lj,...mj,...mab,...lb->...ka", i_reduced, mu_sqrt, v, v, nabla2_psi_theta, nabla2_psi_theta_inv_grad)

    # repul_term1: [..., K, D]
    repul_term1 = 1 / (K * K) * tf.einsum(
        "...km,...j,...ljd,...mj,...mab,...lbd->...ka", i_reduced, mu_sqrt, v_theta_grad, v, nabla2_psi_theta, nabla2_psi_theta_inv)
    # repul_term2: [..., K, D]
    repul_term2 = 1 / (K * K) * tf.einsum(
        "...km,...j,...lj,...mj,...mab,...lbd->...ka", i_reduced, mu_sqrt, v, v, nabla2_psi_theta, grad_nabla2_psi_theta_inv_diag)
    repulsive_term = repul_term1 + repul_term2

    if md:
        # Guarantee mirror descent as a special case when K = 1
        term31 = tf.einsum("...lj,...lia,...lj->...lija", v, v_theta_grad, v)
        term32 = tf.einsum("...lj,...li,...lja->...lija", v, v, v_theta_grad)
        term33 = tf.einsum("...lj,...li,...lj,...lab,...lbd->...lija", v, v, v, nabla2_psi_theta, grad_nabla2_psi_theta_inv_diag)
        repul_grad_term3 = 1 / K * (term31 + term32 - term33)
        # repulsive_term: [..., K, K, D]
        repul_term3 = tf.einsum("...i,...j,...ki,...lija->...kla", mu_sqrt, mu_sqrt, v, repul_grad_term3)
        # repulsive_term: [..., K, D]
        repulsive_term += tf.reduce_mean(repul_term3, axis=-2)

    return weighted_grad + repulsive_term


@tf.function
def truncate_and_grad(eigval, eigvec, n_eigen_threshold, Kxz, grad_Kxz):
    K = eigvec.shape[-2]
    eigen_arr = tf.reduce_mean(tf.reshape(eigval, [-1, K]), axis=0)
    eigen_arr = tf.reverse(eigen_arr, axis=[-1])
    eigen_arr /= tf.reduce_sum(eigen_arr)
    eigen_cum = tf.cumsum(eigen_arr, axis=-1)
    n_eigen = tf.reduce_sum(
        tf.cast(tf.less(eigen_cum, n_eigen_threshold), tf.int32)) + 1
    # eigval: [..., n_eigen]
    # eigvec: [..., K, n_eigen]
    eigval = eigval[..., -n_eigen:]
    eigvec = eigvec[..., -n_eigen:]
    # mu: [..., n_eigen]
    mu = eigval / K
    # v: [..., K, n_eigen]
    v = eigvec * np.sqrt(K)
    # use nystrom formula to fix the gradient
    # v_theta: [..., K, n_eigen]
    # v_theta_grad: [..., K, n_eigen, D]
    v_theta, v_theta_grad = nystrom(Kxz, eigval, eigvec, grad_Kxz=grad_Kxz)
    return mu, v, v_theta, v_theta_grad


def eigen_quantities(theta_, theta, kernel, n_eigen_threshold, jitter=1e-5, kernel_width2=None):
    K = theta.shape[-2]
    # gram: [..., K, K]
    # gram, kernel_width2 = kernel(theta_, theta_)
    Kxz, grad_Kxz, *_ = kernel(theta, theta_, left_grad=True, kernel_width2=kernel_width2)
    gram = tf.stop_gradient(Kxz)
    # tf.debugging.check_numerics(gram, "Gram matrix inf/nan")
    gram = gram + jitter * tf.eye(K, dtype=tf.float64)
#     with tf.device("CPU:0"):
    # try:
    # eigval: [..., K]
    # eigvec: [..., K, K]
    with tf.device("/cpu:0"):
        eigval, eigvec = tf.linalg.eigh(gram)
    # eigval, eigvec = tf.py_function(func=lambda x: np.linalg.eigh(x), inp=[gram], Tout=[tf.float64, tf.float64])
    # except Exception as error:
    #     print(gram)
    #     raise error
    return truncate_and_grad(eigval, eigvec, n_eigen_threshold, Kxz, grad_Kxz)


@tf.function
def svmd_grad(mirror_map, nabla2_psi_theta_inv_grad, theta, theta_, mu, v, v_theta, v_theta_grad):
    K = theta.shape[-2]

    # nabla2_psi_theta: [..., K, D, D]
    nabla2_psi_theta = mirror_map.nabla2_psi(theta_)
    # nabla2_psi_theta_inv: [..., K, D, D]
    nabla2_psi_theta_inv = mirror_map.nabla2_psi_inv(theta)
    # grad_nabla2_psi_inv: [..., K, D, D]
    grad_nabla2_psi_theta_inv_diag = mirror_map.grad_nabla2_psi_inv_diag(theta)

    mu_sqrt = tf.sqrt(mu)
    # [..., K, K]
    i_reduced = tf.einsum("...i,...ki,...mi->...km", mu_sqrt, v, v)
    # weighted_grad: [..., K, D]
    weighted_grad = 1 / (K * K) * tf.einsum(
        "...km,...j,...lj,...mj,...mab,...lb->...ka", i_reduced, mu_sqrt, v_theta, v, nabla2_psi_theta, nabla2_psi_theta_inv_grad)

    # repul_term1: [..., K, D]
    repul_term1 = 1 / (K * K) * tf.einsum(
        "...km,...j,...ljd,...mj,...mab,...lbd->...ka", i_reduced, mu_sqrt, v_theta_grad, v, nabla2_psi_theta, nabla2_psi_theta_inv)
    # repul_term2: [..., K, D]
    repul_term2 = 1 / (K * K) * tf.einsum(
        "...km,...j,...lj,...mj,...mab,...lbd->...ka", i_reduced, mu_sqrt, v_theta, v, nabla2_psi_theta, grad_nabla2_psi_theta_inv_diag)
    # repulsive_term: [..., K, D]
    repulsive_term = repul_term1 + repul_term2

    return weighted_grad + repulsive_term


def svmd_update_v2(target, theta, kernel=imq, n_eigen_threshold=0.98, md=False, kernel_width2=None):
    # Deprecated: Use svmd_update which is faster
    # nabla2_psi_theta_inv_grad: [..., K, D]
    nabla2_psi_theta_inv_grad = target.nabla_psi_inv_grad_logp(theta)
    theta_ = tf.stop_gradient(theta)
    # mu: [..., n_eigen]
    # v, v_theta: [..., K, n_eigen]
    # v_theta_grad: [..., K, n_eigen, D]
    mu, v, v_theta, v_theta_grad = eigen_quantities(theta_, theta, kernel, n_eigen_threshold, kernel_width2=kernel_width2)
    return svmd_grad(target.mirror_map, nabla2_psi_theta_inv_grad, theta, theta_, mu, v, v_theta, v_theta_grad)
