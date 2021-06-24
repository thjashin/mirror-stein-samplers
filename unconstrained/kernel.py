import tensorflow as tf


def compute_squared_dist(x1, x2):
    x1 = x1[:, None, :]
    x2 = x2[None, :, :]
    return tf.reduce_sum(tf.square(x1 - x2), axis=-1)


def heuristic_kernel_width(squared_dists):
    # square_dists: [K1, K2]
    # return: []
    k = tf.shape(squared_dists)[0] * tf.shape(squared_dists)[1] // 2
    top_k_values = tf.nn.top_k(
        tf.reshape(squared_dists, [-1]), k=k, sorted=False).values
    kernel_width2 = tf.reduce_min(top_k_values) #/ 2.
    kernel_width2 = tf.where(
        tf.equal(kernel_width2, 0), tf.ones_like(kernel_width2), kernel_width2)
#     tf.print("kernel_width2:", kernel_width2)
#     tf.print("top_k_values:", tf.reduce_max(top_k_values), tf.reduce_min(top_k_values))
    return tf.stop_gradient(kernel_width2)


def rbf(x1, x2, kernel_width2=None, left_grad=False):
    # x1: [K, D]
    # x2: [K, D]
    ret = []
    squared_dists = compute_squared_dist(x1, x2)
    return_width = False
    if kernel_width2 is None:
        kernel_width2 = heuristic_kernel_width(squared_dists)
        return_width = True
    # k: [K, K]
    k = tf.exp(-squared_dists / kernel_width2)
    ret.append(k)
    if left_grad:
        # grad_x1: [K, K, D]
        grad_x1 = -2 * (x1[:, None, :] - x2[None, :, :]) / kernel_width2 * k[:, :, None]
        ret.append(grad_x1)
#         grad_x1_check = tape.batch_jacobian(k, x1)
#         tf.print("grad_x1:", grad_x1)
#         tf.print("grad_x1_check:", grad_x1_check)
    if return_width:
        ret.append(kernel_width2)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def imq(x1, x2, kernel_width2=None, left_grad=False):
    # x1: [K, D]
    # x2: [K, D]
    ret = []
    squared_dists = compute_squared_dist(x1, x2)
    return_width = False
    if kernel_width2 is None:
        kernel_width2 = heuristic_kernel_width(squared_dists)
        return_width = True
    inner = 1. + squared_dists / kernel_width2
    k = 1. / tf.sqrt(inner)
    ret.append(k)
    if left_grad:
        # grad_x1: [K, K, D]
        grad_x1 = -(inner**(-1.5))[:, :, None] * (x1[:, None, :] - x2[None, :, :]) / kernel_width2 
        ret.append(grad_x1)
#         grad_x1_check = tape.batch_jacobian(k, x1)
#         tf.print("grad_x1:", grad_x1)
#         tf.print("grad_x1_check:", grad_x1_check)
    if return_width:
        ret.append(kernel_width2)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def nystrom(z, x, eigval, eigvec, kernel_width2=None, kernel=rbf, grad=False):
    # z: [M, D]
    # x: [N, D]
    # eigvec: [M, n_eigen]
    # eigval: [n_eigen]
    # Kxz: [N, M], grad_Kxz: [N, M, D]
    K = tf.cast(tf.shape(z)[0], z.dtype)
    if grad:
        Kxz, grad_Kxz = kernel(x, z, kernel_width2=kernel_width2, left_grad=True)
    else:
        Kxz = kernel(x, z, kernel_width2=kernel_width2)
    # ret: [N, n_eigen]
    u = tf.sqrt(K) * tf.matmul(Kxz, eigvec) / eigval
    if grad:
        # grad_u: [N, n_eigen, D]
        grad_u = tf.sqrt(K) * tf.einsum("nml,mj->njl", grad_Kxz, eigvec) / eigval[None, :, None]
        return u, grad_u
    else:
        return u
