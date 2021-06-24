# Adapted from https://github.com/dilinwang820/matrix_svgd/blob/master/lr/gmm_models.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
import numpy as np


def _sum_log_exp(X, mus, dcovs, weights):

    dim = tf.cast(tf.shape(mus)[1], tf.float32)
    _lnD = tf.reduce_sum(tf.math.log(dcovs), axis=1)

    diff = tf.expand_dims(X, 0) - tf.expand_dims(mus, 1)  # c x n x d
    diff_times_inv_cov = diff * tf.expand_dims(1./ dcovs, 1)  # c x n x d
    sum_sq_dist_times_inv_cov = tf.reduce_sum(diff_times_inv_cov * diff, axis=2)  # c x n 
    ln2piD = tf.math.log(2 * np.pi) * dim
    #log_coefficients = tf.expand_dims(ln2piD + tf.log(self._D), 1) # c x 1
    log_coefficients = tf.expand_dims(ln2piD + _lnD, 1) # c x 1
    log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)  # c x n
    log_weighted = log_components + tf.expand_dims(tf.math.log(weights), 1)  # c x n + c x 1
    log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)

    return log_weighted, log_shift



def _log_gradient(X, mus, dcovs, weights):  

    # X: n_samples x d; mu: c x d; cov: c x d x d
    x_shape = X.get_shape()
    assert len(list(x_shape)) == 2, 'illegal inputs'

    def posterior(X):
        log_weighted, log_shift = _sum_log_exp(X, mus, dcovs, weights)
        prob = tf.exp(log_weighted - log_shift) # c x n
        prob = prob / tf.reduce_sum(prob, axis=0, keepdims=True)
        return prob

    diff = tf.expand_dims(X, 0) - tf.expand_dims(mus, 1)  # c x n x d
    diff_times_inv_cov = -diff * tf.expand_dims(1./dcovs, 1)  # c x n x d

    P = posterior(X)  # c x n
    score = tf.matmul(
        tf.expand_dims(tf.transpose(P, perm=[1, 0]), 1), # n x 1 x c
        tf.transpose(diff_times_inv_cov, perm=[1, 0, 2]) # n x c x d
    ) 
    return tf.squeeze(score) # n x d


def mixture_weights_and_grads(X, mus=None, dcovs=None, weights=None):  
    # X: n_samples x d; 
    # x_shape = X.get_shape()
    x_shape = X.shape
    assert len(list(x_shape)) == 2, 'illegal inputs'

    if mus is None:
        mus = tf.stop_gradient(X)
    if dcovs is None:
        dcovs = tf.cast(tf.ones_like(mus), tf.float32)
    # uniform weights, only care about ratio
    if weights is None: 
        weights = tf.ones(tf.shape(mus)[0])

    log_weighted, log_shift = _sum_log_exp(X, mus, dcovs, weights)
    exp_log_shifted = tf.exp(log_weighted - log_shift) # c x n
    exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, axis=0, keepdims=True) # 1 x n
    p = exp_log_shifted / exp_log_shifted_sum

    # weights
    mix = tf.transpose(p)  # n * c
    d_log_gmm = _log_gradient(X, mus, dcovs, weights) # n * d

    d_log_gau = -(tf.expand_dims(X, 1) - tf.expand_dims(mus, 0)) / tf.expand_dims(dcovs, 0) # n x c x d
    mix_grad = d_log_gau - tf.expand_dims(d_log_gmm, 1)

    # c * n, c * n * d
    return tf.transpose(mix), tf.transpose(mix_grad, [1,0,2])



#from models import GaussianMixture
#
#def _simulate_mixture_target(n_components=10, dim = 1, val=5., seed=123):
#
#    with tf.variable_scope('p_target') as scope:
#        np.random.seed(seed)
#        mu0 = tf.get_variable('mu', initializer=np.random.uniform(-val, val, size=(n_components, dim)).astype('float32'), dtype=tf.float32,  trainable=False)
#
#        log_sigma0 = tf.zeros((n_components, dim))
#        weights0 = tf.ones(n_components) / n_components
#        p_target = GaussianMixture(n_components, mu0, log_sigma0, weights0)
#
#        return p_target
#
#
#
#if __name__ == '__main__':
#
#    session_config = tf.ConfigProto(
#        allow_soft_placement=True,
#        gpu_options=tf.GPUOptions(allow_growth=True),
#    )
#
#    from ops import rbf_kernel
#    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
#
#        x_train = tf.constant(np.random.normal(size=(3, 2)).astype('float32'))
#        k1, dk1 = rbf_kernel(x_train)
#        k2, dk2 = rbf_kernel(x_train, to3d=True)
#
#        p_target = _simulate_mixture_target(n_components=3, dim=2, val=1.0)
#        Hs = tf.hessians(p_target.log_prob(x_train), tf.split(x_train, num_or_size_splits=3, axis=0))
#
#        tf.global_variables_initializer().run()
#
#        hs = sess.run([Hs])
#        print(hs.shape)
#        
#        #dxk1, dxk2 = sess.run([dk1, dk2])
#        #print (dxk1)
#        #print (np.sum(dxk2, 0))
#        #k1, dk1 = rbf_kernel(x_train)
#
