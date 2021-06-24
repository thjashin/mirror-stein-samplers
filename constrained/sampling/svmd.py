import tensorflow as tf
import numpy as np

from .kernel import nystrom, rbf
# from constrained.nonnegative.entropic import NonnegativeEntropicMap, safe_reciprocal


def compute_K_psi(target, theta, kernel, n_eigen_threshold, md=False, jitter=1e-5):
    K = theta.shape[0]
    theta_ = tf.stop_gradient(theta)
    # gram: [K, K]
    gram, kernel_width2 = kernel(theta_, theta_)
    tf.debugging.check_numerics(gram, "Gram matrix inf/nan")
    gram = gram + jitter * tf.eye(K, dtype=tf.float64)
#     kernel_width = tf.constant(0.01, dtype=tf.float64)
#     gram = rbf(theta_, theta_, kernel_width=kernel_width)
    # eigval: [K]
    # eigvec: [K, K]
#     print(theta_)
#     print(kernel_width)
#     with tf.device("CPU:0"):
    try:
        eigval, eigvec = tf.linalg.eigh(gram)
#         tf.print("gram:", gram)
    except Exception as error:
        print(gram)
        raise error
#     tf.print("eigval:", eigval)
    if n_eigen_threshold is not None:
        eigen_arr = tf.reverse(eigval, axis=[-1])
        eigen_arr /= tf.reduce_sum(eigen_arr)
        eigen_cum = tf.cumsum(eigen_arr, axis=-1)
        n_eigen = tf.reduce_sum(
            tf.cast(tf.less(eigen_cum, n_eigen_threshold), tf.int32)) + 1
#         tf.print("n_eigen:", n_eigen)
        # eigval: [n_eigen]
        # eigvec: [K, n_eigen]
        eigval = eigval[-n_eigen:]
        eigvec = eigvec[:, -n_eigen:]
    # mu: [n_eigen]
    mu = eigval / K
#     tf.print("mu:", mu)
    # v: [K, n_eigen]
    v = eigvec * np.sqrt(K)
    # use nystrom formula to fix the gradient
    # v_theta: [K, n_eigen]
    # v_theta_grad: [K, n_eigen, D]
    v_theta, v_theta_grad = nystrom(
        theta_, theta, eigval, eigvec, kernel_width2=kernel_width2, kernel=kernel, grad=True)
    if md:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            # nabla2_psi_theta: [K, D, D]
            nabla2_psi_theta = target.mirror_map.nabla2_psi(theta)
        # grad_nabla2_psi: [K, D, D, D]
        grad_nabla2_psi = tape.batch_jacobian(nabla2_psi_theta, theta)
        # Gamma: [n_eigen, n_eigen, D, D]
        Gamma = 1 / K * tf.einsum("ki,kj,kab->ijab", v_theta, v_theta, nabla2_psi_theta)
    else:
        # nabla2_psi_theta: [K, D, D]
        nabla2_psi_theta = target.mirror_map.nabla2_psi(theta_)
        # Gamma: [n_eigen, n_eigen, D, D]
        Gamma = 1 / K * tf.einsum("ki,kj,kab->ijab", v, v, nabla2_psi_theta)
    
#     print("v:", v)
#     print("v_theta:", v_theta)
    mu_sqrt = tf.sqrt(mu)
    # K_psi: [K, K, D, D]
    K_psi = tf.einsum("i,j,ki,lj,ijab->klab", mu_sqrt, mu_sqrt, v, v_theta, Gamma)
    # nabla2_psi_theta_inv: [K, D, D]
    nabla2_psi_theta_inv = target.mirror_map.nabla2_psi_inv(theta)
    # repul_grad_term1: [K, n_eigen, n_eigen, D]
    repul_grad_term1 = tf.einsum("ljd,ijab,lbd->lija", v_theta_grad, Gamma, nabla2_psi_theta_inv)
    # grad_nabla2_psi_inv: [K, D, D]
    grad_nabla2_psi_theta_inv_diag = target.mirror_map.grad_nabla2_psi_inv_diag(theta)
    # repul_grad_term2: [K, n_eigen, n_eigen, D]
    repul_grad_term2 = tf.einsum("lj,ijab,lbd->lija", v_theta, Gamma, grad_nabla2_psi_theta_inv_diag)
    repul_grad = repul_grad_term1 + repul_grad_term2
    if md:
        # TODO: can be simplified since the last two terms will reduce to identity
        term31 = tf.einsum("lj,lid,lj,lab,lbd->lija", 
                            v_theta, v_theta_grad, v_theta, nabla2_psi_theta, nabla2_psi_theta_inv)
        term32 = tf.einsum("lj,li,ljd,lab,lbd->lija",
                            v_theta, v_theta, v_theta_grad, nabla2_psi_theta, nabla2_psi_theta_inv)
        term33 = tf.einsum("lj,li,lj,labd,lbd->lija",
                            v_theta, v_theta, v_theta, grad_nabla2_psi, nabla2_psi_theta_inv)
        repul_grad_term3 = 1 / K * (term31 + term32 + term33)
        repul_grad += repul_grad_term3
    # repulsive_term: [K, K, D]
    repulsive_term = tf.einsum("i,j,ki,lija->kla", mu_sqrt, mu_sqrt, v, repul_grad)
    # repulsive_term: [K, D]
    repulsive_term = tf.reduce_mean(repulsive_term, axis=1)
    return K_psi, repulsive_term


@tf.function
def svmd_update(target, theta, kernel=rbf, n_eigen_threshold=0.98, md=False):
    K = theta.shape[0]
    K_psi, repulsive_term = compute_K_psi(target, theta, kernel, n_eigen_threshold, md=md)
    # nabla2_psi_theta_inv_grad: [K, D]
    nabla2_psi_theta_inv_grad = target.nabla_psi_inv_grad_logp(theta)
    # weighted_grad: [K, D]
    weighted_grad = 1 / K * tf.einsum("klab,lb->ka", K_psi, nabla2_psi_theta_inv_grad)
#     print(weighted_grad.dtype, repulsive_term.dtype)
#     tf.print("grad_term:", weighted_grad)
#     tf.print("repul_term:", repulsive_term)
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


def eigen_quantities(theta_, theta, kernel, n_eigen_threshold, jitter=1e-5):
    K = theta.shape[-2]
    # gram: [..., K, K]
    # gram, kernel_width2 = kernel(theta_, theta_)
    Kxz, grad_Kxz, _ = kernel(theta, theta_, left_grad=True)
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


# @tf.function(experimental_compile=True)
# def svmd_grad_nonnegative_entropic(target, theta, theta_, mu, v, v_theta, v_theta_grad):
#     K = theta.shape[-2]
#     nabla2_psi_theta_inv_grad = target.nabla_psi_inv_grad_logp(theta)

#     mu_sqrt = tf.sqrt(mu)
#     i_reduced = tf.einsum("...i,...ki,...mi->...km", mu_sqrt, v, v)

#     theta_rec_bc = tf.repeat(safe_reciprocal(theta)[..., None, :], K, axis=-2)
#     theta_bc = tf.repeat(theta[..., None, :, :], K, axis=-3)
#     repul_term1 = 1 / (K * K) * tf.einsum(
#         "...km,...j,...lja,...mj,...mla,...mla->...ka", i_reduced, mu_sqrt, v_theta_grad, v, theta_rec_bc, theta_bc)

#     repul_term2_pre = 1 / (K * K) * tf.einsum(
#         "...km,...j,...lj,...mj,...mla->...kla", i_reduced, mu_sqrt, v_theta, v, theta_rec_bc)
#     repul_term2 = tf.reduce_sum(repul_term2_pre, axis=-2)
#     repulsive_term = repul_term1 + repul_term2

#     weighted_grad = tf.einsum(
#         "...kla,...la->...ka", repul_term2_pre, nabla2_psi_theta_inv_grad)

#     return weighted_grad + repulsive_term


@tf.function(experimental_compile=True)
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


def svmd_update_v2(target, theta, kernel=rbf, n_eigen_threshold=0.98, md=False):
    # nabla2_psi_theta_inv_grad: [..., K, D]
    nabla2_psi_theta_inv_grad = target.nabla_psi_inv_grad_logp(theta)
    theta_ = tf.stop_gradient(theta)
    # mu: [..., n_eigen]
    # v, v_theta: [..., K, n_eigen]
    # v_theta_grad: [..., K, n_eigen, D]
    mu, v, v_theta, v_theta_grad = eigen_quantities(theta_, theta, kernel, n_eigen_threshold)
    return svmd_grad(target.mirror_map, nabla2_psi_theta_inv_grad, theta, theta_, mu, v, v_theta, v_theta_grad)
