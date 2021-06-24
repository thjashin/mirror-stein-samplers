import tensorflow as tf

from .kernel import rbf, imq


def sqr_dist(x, y, e=1e-8):
    xx = tf.reduce_sum(tf.square(x) + 1e-10, axis=1)
    yy = tf.reduce_sum(tf.square(y) + 1e-10, axis=1)
    xy = tf.matmul(x, y, transpose_b=True)
    dist = tf.expand_dims(xx, 1) + tf.expand_dims(yy, 0) - 2. * xy
    return dist


def median_distance(H):
    V = tf.reshape(H, [-1])
    n = tf.size(V)
    top_k, _ = tf.nn.top_k(V, k=(n // 2) + 1)
    return top_k[-1]


def rbf_kernel(x, h=-1, to3d=False):
    H = sqr_dist(x, x)
    if h == -1:
        h = tf.maximum(tf.constant(1e-6, x.dtype), median_distance(H))

    kxy = tf.exp(-H / h)
    dxkxy = -tf.matmul(kxy, x)
    sumkxy = tf.reduce_sum(kxy, axis=1, keepdims=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    if to3d: 
        dxkxy = -(tf.expand_dims(x, 1) - tf.expand_dims(x, 0)) * tf.expand_dims(kxy, 2) * 2. / h
    return kxy, dxkxy


def imq_kernel(x, h=-1):
    H = sqr_dist(x, x)
    if h == -1:
        h = median_distance(H)

    kxy = 1. / tf.sqrt(1. + H / h) 

    dxkxy = .5 * kxy / (1. + H / h)
    dxkxy = -tf.matmul(dxkxy, x)
    sumkxy = tf.reduce_sum(kxy, axis=1, keepdims=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy


def svgd_update(x, grad, kernel='rbf', temperature=1., u_kernel=None, **kernel_params):
    assert x.shape[1:] == grad.shape[1:], 'illegal inputs and grads'
    if len(x.shape) > 2:
        x = tf.reshape(x, (x.shape[0], -1))
        grad = tf.reshape(grad, (grad.shape[0], -1))

    if u_kernel is not None:
        kxy, dxkxy = u_kernel['kxy'], u_kernel['dxkxy']
        dxkxy = tf.reshape(dxkxy, x.shape)
    else:
        if kernel == rbf:
            kxy, dxkxy = rbf_kernel(x, **kernel_params)
        elif kernel == imq:
            kxy, dxkxy = imq_kernel(x)
        elif kernel == 'none':
            kxy = tf.eye(x.shape[0], dtype=x.dtype)
            dxkxy = tf.zeros_like(x)
        else:
            raise NotImplementedError

    svgd_grad = (tf.matmul(kxy, grad) + temperature * dxkxy) / tf.reduce_sum(kxy, axis=1, keepdims=True)
    svgd_grad = tf.reshape(svgd_grad, x.shape)
    return svgd_grad
