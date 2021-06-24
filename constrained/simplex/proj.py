import tensorflow as tf


@tf.function
def euclidean_proj_simplex(v, s=1):
    """ 
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (..., n) numpy array, n-dimensional vector to project
    s: int, optional, default: 1, radius of the simplex

    Returns
    -------
    w: (n,) numpy array, Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

    [2] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n = v.shape[-1]  # will raise ValueError if v is not 1-D
    batch_shape = v.shape[:-1]
    v = tf.reshape(v, [-1, n])
    # check if we are already on the simplex
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = tf.sort(v, axis=-1, direction="DESCENDING")
    cssv = tf.cumsum(u, axis=-1)
    j = tf.range(1, n + 1, dtype=u.dtype)
    rho = tf.reduce_sum(tf.cast(u * j - cssv + s > 0, u.dtype), axis=-1, keepdims=True) - 1.
    # max_nn = cssv[tf.range(v.shape[0]), rho[:, 0]]
    max_nn = tf.gather_nd(cssv, tf.stack([tf.range(v.shape[0]), tf.cast(rho[:, 0], tf.int32)], axis=1))
    theta = (max_nn[:, None] - s) / (rho + 1)
    w = tf.maximum(v - theta, 0.)
    w = tf.reshape(w, tf.concat([batch_shape, [n]], -1))
    return w
