import tensorflow as tf

from .kernel import imq, nystrom


def compute_K_psi(theta, G, G_inv, grad_G_inv, kernel, n_eigen_threshold, md=False, jitter=1e-5, grad_G_inv_trace=None, use_jitter=False):
    K = tf.cast(tf.shape(theta)[0], theta.dtype)
    theta_ = tf.stop_gradient(theta)
    # gram: [K, K]
    gram, kernel_width2 = kernel(theta_, theta_)
    if use_jitter:
        gram = gram + jitter * tf.eye(K, dtype=gram.dtype)
    # eigval: [K]
    # eigvec: [K, K]
    # with tf.device("CPU:0"):
    eigval, eigvec = tf.linalg.eigh(gram)
    if n_eigen_threshold is not None:
        eigen_arr = tf.reverse(eigval, axis=[-1])
        eigen_arr /= tf.reduce_sum(eigen_arr)
        eigen_cum = tf.cumsum(eigen_arr, axis=-1)
        n_eigen = tf.reduce_sum(
            tf.cast(tf.less(eigen_cum, n_eigen_threshold), tf.int32)) + 1
        # eigval: [n_eigen]
        # eigvec: [K, n_eigen]
        eigval = eigval[-n_eigen:]
        eigvec = eigvec[:, -n_eigen:]
    # mu: [n_eigen]
    mu = eigval / K
    # v: [K, n_eigen]
    v = eigvec * tf.sqrt(K)
    # use nystrom formula to fix the gradient
    # v_theta: [K, n_eigen]
    # v_theta_grad: [K, n_eigen, D]
    v_theta, v_theta_grad = nystrom(
        theta_, theta, eigval, eigvec, kernel_width2=kernel_width2, kernel=kernel, grad=True)
    if md:
        raise NotImplementedError("grad_G should be a variable, estimated by moving average.")
        # grad_G: [K, D, D, D]
        grad_G = batch_jacobian(G, theta)
        # G: [K, D, D]
        # Gamma: [n_eigen, n_eigen, D, D]
        # TODO: can use v instead?
        Gamma = 1. / K * tf.einsum("ki,kj,kab->ijab", v_theta, v_theta, G)
    else:
        # G: [K, D, D]
        # Gamma: [n_eigen, n_eigen, D, D]
        Gamma = 1. / K * tf.einsum("ki,kj,kab->ijab", v, v, G)

    mu_sqrt = tf.sqrt(mu)
    # K_psi: [K, K, D, D]
    K_psi = tf.einsum("i,j,ki,lj,ijab->klab", mu_sqrt, mu_sqrt, v, v_theta, Gamma)
    # repul_grad_term1: [K, n_eigen, n_eigen, D]
    repul_grad_term1 = tf.einsum("ljd,ijab,lbd->lija", v_theta_grad, Gamma, G_inv)
    if grad_G_inv_trace is not None:
        repul_grad_term2 = tf.einsum("lj,ijab,lb->lija", v_theta, Gamma, grad_G_inv_trace)
    else:
        # grad_G_inv: [K, D, D]
        # repul_grad_term2: [K, n_eigen, n_eigen, D]
        repul_grad_term2 = tf.einsum("lj,ijab,lbd->lija", v_theta, Gamma, grad_G_inv)
    repul_grad = repul_grad_term1 + repul_grad_term2
    if md:
        # TODO: can be simplified since the last two terms will reduce to identity
        term31 = tf.einsum("lj,lid,lj,lab,lbd->lija", 
                            v_theta, v_theta_grad, v_theta, G, G_inv)
        term32 = tf.einsum("lj,li,ljd,lab,lbd->lija",
                            v_theta, v_theta, v_theta_grad, G, G_inv)
        term33 = tf.einsum("lj,li,lj,labd,lbd->lija",
                            v_theta, v_theta, v_theta, grad_G, G_inv)
        repul_grad_term3 = 1 / K * (term31 + term32 + term33)
        repul_grad += repul_grad_term3
    # repulsive_term: [K, K, D]
    repulsive_term = tf.einsum("i,j,ki,lija->kla", mu_sqrt, mu_sqrt, v, repul_grad)
    # repulsive_term: [K, D]
    repulsive_term = tf.reduce_mean(repulsive_term, axis=1)
    return K_psi, repulsive_term


def svmd_update(theta, theta_grads, G, G_inv, grad_G_inv=None, kernel=imq, n_eigen_threshold=0.98, md=False, grad_G_inv_trace=None, G_inv_grads=None, use_jitter=False):
    K = tf.cast(tf.shape(theta)[0], theta.dtype)
    # theta: [K, D]
    # theta_grads: [K, D]
    # G, G_inv: [K, D, D]
    # grad_G_inv: [K, D, D]
    K_psi, repulsive_term = compute_K_psi(
        theta, G, G_inv, grad_G_inv, kernel, n_eigen_threshold, md=md, grad_G_inv_trace=grad_G_inv_trace, use_jitter=use_jitter)
    if theta_grads is None:
        G_inv_grad = G_inv_grads
    else:
        # G_inv_grad: [K, D]
        G_inv_grad = tf.squeeze(G_inv @ theta_grads[:, :, None], -1)
    # weighted_grad: [K, D]
    weighted_grad = 1. / K * tf.einsum("klab,lb->ka", K_psi, G_inv_grad)
#     print(weighted_grad.dtype, repulsive_term.dtype)
#     tf.print("grad_term:", weighted_grad)
#     tf.print("repul_term:", repulsive_term)
    return weighted_grad + repulsive_term
