import tensorflow as tf
import numpy as np


@tf.function
def cross_diff(x1, x2):
    x1 = x1[..., :, None, :]
    x2 = x2[..., None, :, :]
    return x1 - x2


@tf.function(experimental_compile=True)
def compute_squared_dist(cross_diff_val):
    return tf.reduce_sum(tf.square(cross_diff_val), axis=-1)


@tf.function
def heuristic_kernel_width(squared_dists):
    # square_dists: [..., K1, K2]
    # return: [...]
    n_elems = squared_dists.shape[-1] * squared_dists.shape[-2]
    top_k_values = tf.math.top_k(
        tf.reshape(squared_dists, [-1, n_elems]), k=n_elems // 2, sorted=False).values
    kernel_width2 = tf.reduce_min(top_k_values, axis=-1) #/ 2.
    kernel_width2 = tf.where(
        tf.equal(kernel_width2, 0), tf.ones_like(kernel_width2), kernel_width2)
    kernel_width2 = tf.reshape(kernel_width2, squared_dists.shape[:-2])
#     tf.print("kernel_width2:", kernel_width2)
#     tf.print("top_k_values:", tf.reduce_max(top_k_values), tf.reduce_min(top_k_values))
    return tf.stop_gradient(kernel_width2)


@tf.function
def rbf(x1, x2, kernel_width2=None, left_grad=False):
    # x1: [..., K, D]
    # x2: [..., K, D]
    ret = []
    cross_x1_x2 = cross_diff(x1, x2)
    squared_dists = compute_squared_dist(cross_x1_x2)
    return_width = False
    if kernel_width2 is None:
        kernel_width2 = heuristic_kernel_width(squared_dists)
        return_width = True
    else:
        kernel_width2 = tf.constant(kernel_width2, dtype=x1.dtype)
    # k: [..., K, K]
    k = tf.exp(-squared_dists / kernel_width2[..., None, None])
    ret.append(k)
    if left_grad:
        # grad_x1: [..., K, K, D]
        grad_x1 = -2 * cross_x1_x2 / kernel_width2[..., None, None, None] * k[..., None]
        ret.append(grad_x1)
#         grad_x1_check = tape.batch_jacobian(k, x1)
#         tf.print("grad_x1:", grad_x1)
#         tf.print("grad_x1_check:", grad_x1_check)
    if return_width:
        ret.append(kernel_width2)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


@tf.function
def imq(x1, x2, kernel_width2=None, left_grad=False):
    # x1: [..., K, D]
    # x2: [..., K, D]
    ret = []
    cross_x1_x2 = cross_diff(x1, x2)
    squared_dists = compute_squared_dist(cross_x1_x2)
    return_width = False
    if kernel_width2 is None:
        kernel_width2 = heuristic_kernel_width(squared_dists)
        return_width = True
    else:
        kernel_width2 = tf.constant(kernel_width2, dtype=x1.dtype)
    inner = 1. + squared_dists / kernel_width2[..., None, None]
    k = 1. / tf.sqrt(inner)
    ret.append(k)
    if left_grad:
        # grad_x1: [K, K, D - 1]
        grad_x1 = -(inner**(-1.5))[..., None] * cross_x1_x2 / kernel_width2[..., None, None, None]
        ret.append(grad_x1)
#         grad_x1_check = tape.batch_jacobian(k, x1)
#         tf.print("grad_x1:", grad_x1)
#         tf.print("grad_x1_check:", grad_x1_check)
    if return_width:
        ret.append(kernel_width2)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


@tf.function(experimental_compile=True)
def nystrom(Kxz, eigval, eigvec, grad_Kxz=None):
    # Kxz: [..., N, M], grad_Kxz: [..., N, M, D]
    # eigvec: [..., M, n_eigen]
    # eigval: [..., n_eigen]
    M = Kxz.shape[-1]
    # # z: [..., M, D]
    # # x: [..., N, D]
    # if grad:
    #     Kxz, grad_Kxz = kernel(x, z, kernel_width2=kernel_width2, left_grad=True)
    # else:
    #     Kxz = kernel(x, z, kernel_width2=kernel_width2)
    # ret: [..., N, n_eigen]
    u = np.sqrt(M) * tf.matmul(Kxz, eigvec) / eigval[..., None, :]
    if grad_Kxz is not None:
        # grad_u: [..., N, n_eigen, D]
        grad_u = np.sqrt(M) * tf.einsum("...nml,...mj->...njl", grad_Kxz, eigvec) / eigval[..., None, :, None]
        return u, grad_u
    else:
        return u
