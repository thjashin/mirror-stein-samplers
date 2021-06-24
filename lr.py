import os
import time
import gc

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import scipy
import scipy.io
import pandas as pd
from absl import app
from absl import flags

tf.random.set_seed(1234)
np.random.seed(1234)

FLAGS = flags.FLAGS
# flags.DEFINE_float("lr", 0.1, 'Learning rate.')

from unconstrained.svmd import svmd_update
from unconstrained.svgd import svgd_update, rbf_kernel
from unconstrained.rsvgd import rsvgd_update
from unconstrained.kernel import imq, rbf
from lr.gmm_models import mixture_weights_and_grads


class LogisticRegression:
    def __init__(self, n_particles, D):
        self.K = n_particles
        self.D = D
        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initial_value=initializer(shape=[n_particles, D], dtype=tf.float32), dtype=tf.float32)
        self.G = tf.Variable(initial_value=tf.zeros([n_particles, D, D], dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.grad_G = tf.Variable(initial_value=tf.zeros([n_particles, D, D, D], dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.step = tf.Variable(initial_value=1., dtype=tf.float32)


def mixture_gradient(W, W_grads, H_inv, n_particles):
    # for the \ell-th cluster
    def _weighted_svgd(x, d_log_pw, w):
        kxy, dxkxy = rbf_kernel(x, to3d=True)
        velocity = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(w, 0), 2) * tf.expand_dims(kxy, 2) * tf.expand_dims(d_log_pw, 0), axis=1) + \
            tf.reduce_sum(tf.expand_dims(tf.expand_dims(w, 1), 2) * dxkxy, axis=0)
        # n * d , d x d
        return velocity


    def _mixture_svgd_grads(x, d_log_p, mix, mix_grads, H_inv):
        velocity = 0
        for i in range(n_particles):
            w_i_svgd = _weighted_svgd(x, d_log_p + mix_grads[i], mix[i])

            # H_\ell
            delta = tf.matmul(w_i_svgd, H_inv[i])
            velocity += tf.expand_dims(mix[i], 1) * delta
        return  velocity

    mix, mix_grads = mixture_weights_and_grads(W)  # c * n, c * n * d
    velocity = _mixture_svgd_grads(W, W_grads, mix, mix_grads, H_inv)
    return -velocity


@tf.function
def compute_ll_acc(y_prob, y):
    # y_pred: [bs]
    y_pred = tf.reduce_mean(y_prob, 0)
    # ll: [bs]
    ll = y * tf.math.log(y_pred + 1e-8) + (1. - y) * tf.math.log(1. - y_pred + 1e-8)
    # accuracy: [bs]
    accuracy = tf.cast(
        tf.equal(tf.cast(tf.greater(y, 0.5), tf.int32), 
                 tf.cast(tf.greater(y_pred, 0.5), tf.int32)), tf.float32)
    return ll, accuracy


def run(seed, df_rows, model, train, test, valid, N, method="smvd", n_epochs=2, lr=0.1):
    def evaluate(seed, step, df_rows, model, timer):
        def get_lik_and_acc(dataset):
            lls = []
            accs = []
            for batch in dataset:
                x_batch, y_batch = batch
                z = tf.reduce_sum(tf.expand_dims(x_batch, 0) * tf.expand_dims(model.w, 1), -1)
                y_prob = tf.sigmoid(z)
                ll_i, acc_i = compute_ll_acc(y_prob, y_batch)
                lls.append(ll_i.numpy())
                accs.append(acc_i.numpy())
            lls = np.hstack(lls)
            accs = np.hstack(accs)
            return np.mean(lls), np.mean(accs)

        valid_ll, valid_acc = get_lik_and_acc(valid)
        df_rows.append({
            "method": method,
            "lr": lr,
            "seed": seed,
            "step": step,
            "type": "valid",
            "ll": valid_ll,
            "acc": valid_acc,
            "time": timer + time.time(),
        })
        test_ll, test_acc = get_lik_and_acc(test)
        df_rows.append({
            "method": method,
            "lr": lr,
            "seed": seed,
            "step": step,
            "type": "test",
            "ll": test_ll,
            "acc": test_acc,
            "time": timer + time.time(),
        })

    @tf.function
    def train_one_step(model, optimizer, X, y, n_train, method="svmd", kernel=imq):
        # with tf.GradientTape() as tape:
        # X: [B, D]
        # y: [B]
        batch_size = tf.cast(tf.shape(X)[0], tf.float32)
        # z: [K, B]
        z = tf.reduce_sum(tf.expand_dims(X, 0) * tf.expand_dims(model.w, 1), -1)
        # y_prob: [K, B]
        y_prob = tf.sigmoid(z)

        # dz: [K, B]
        dz = y_prob * (1. - y_prob)

        # # empirical fisher:
        # # mean_dW: [K, 1, D]
        # mean_dW = tf.reduce_mean(dW, axis=1, keepdims=True)
        # # diff_dW: [K, B, D]
        # diff_dW = dW - mean_dW
        # # emp_cov_dW_: [K, D, D]
        # emp_cov_dW_ = tf.matmul(diff_dW, diff_dW, transpose_a=True) / batch_size
        # cov_dW = emp_cov_dW_

        # batch_fisher: [K, D, D]
        batch_fisher = tf.einsum("kb,bd,bi->kdi", dz, X, X) / batch_size

        ll, accuracy = compute_ll_acc(y_prob, y)

        # dW: [K, B, D]
        dw = X[None, :, :] * tf.expand_dims(y - y_prob, 2)

        # w_grads: [K, D]
        w_grads = tf.reduce_mean(dw, axis=1) * n_train - model.w

        if method == "svgd":
            grads = svgd_update(model.w, w_grads, kernel=kernel)
        else:
            # update covariance
            rho = tf.minimum(1. - 1. / model.step, 0.95)
            model.G.assign(rho * model.G + (1. - rho) * batch_fisher)

            H = model.G + 1e-2 * tf.eye(model.D, dtype=tf.float32)
            H_inv = tf.linalg.inv(H)

            if method == "matrix_svgd_avg":
                grads = svgd_update(model.w, w_grads, kernel=kernel)
                grads = tf.matmul(grads, tf.reduce_mean(H_inv, 0))
            elif method == "matrix_svgd_mixture":
                grads = -1. * mixture_gradient(model.w, w_grads, H_inv, model.K)
            else:
                # grad_cov_dw; [K, D, D, D]
                # grad_batch_fisher_check = tape.batch_jacobian(batch_fisher, model.w)
                grad_batch_fisher = tf.einsum("kb,kb,bi,bj,bd->kijd", dz, 1. - 2 * y_prob, X, X, X) / batch_size
                model.grad_G.assign(rho * model.grad_G + (1. - rho) * grad_batch_fisher)

                # grad_H_inv: [K, D, D]
                grad_H_inv = tf.linalg.diag_part(-tf.einsum("kab,kbcd,kce->kaed", H_inv, model.grad_G, H_inv))
                if method == "rsvgd":
                    grads = rsvgd_update(model.w, w_grads, H, H_inv, grad_H_inv, kernel=kernel)
                elif method == "svmd":
                    eta_grads = svmd_update(
                        model.w, w_grads, H, H_inv, grad_H_inv, kernel=kernel, n_eigen_threshold=0.98, md=False)
                    grads = tf.squeeze(H_inv @ eta_grads[:, :, None], axis=-1)
                else:
                    raise NotImplementedError()

        optimizer.apply_gradients([(-1. * grads / n_train, model.w)])
        model.step.assign(model.step + 1.)

        return ll, accuracy

    print("learning rate:", lr)
    optimizer = tf.keras.optimizers.SGD(lr)
    step = 1
    timer = -time.time()
    for ep in range(1, n_epochs + 1):
        print("epoch {}/{}".format(ep, n_epochs))
        progbar = tf.keras.utils.Progbar(len(train))
        for batch in train:
            x_batch, y_batch = batch
            train_one_step(model, optimizer, x_batch, y_batch, n_train=N, method=method, kernel=rbf)
            # df_rows.append({
            #     "method": method,
            #     "seed": seed,
            #     "step": step,
            #     "type": "train",
            #     "ll": ll.numpy().mean(),
            #     "acc": acc.numpy().mean(),
            # })
            progbar.add(1)
            if step % 10 == 0:
                evaluate(seed, step, df_rows, model, timer)
            step += 1


def main(argv):
    data = scipy.io.loadmat('lr/covertype.mat')
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = 0
    X_input = np.hstack([X_input, np.ones([len(X_input), 1])]).astype(np.float32)
    y_input = y_input.astype(np.float32)

    n_particles = 20
    batch_size = 256
    eval_bs = 2048
    n_epochs = 2

    nrep = 5
    rng = np.random.RandomState(1)
    seeds = rng.randint(1, 1e6, size=nrep)
    lr_list = [1., 0.5, 0.1, 0.05, 0.01]
    methods = ["rsvgd", "svgd", "svmd", "matrix_svgd_avg", "matrix_svgd_mixture"]
    # methods = ["matrix_svgd_mixture"]
    df_rows = []

    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=seed)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
        print("train:", x_train.shape)
        with tf.device("/cpu:0"):
            x_train = tf.constant(x_train, dtype=tf.float32)
            y_train = tf.constant(y_train, dtype=tf.float32)
        N, D = x_train.shape

        train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True)
        test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(eval_bs)
        valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(eval_bs)

        for lr in lr_list:
            for method in methods:
                print("method:", method)
                model = LogisticRegression(n_particles, D)
                try:
                    run(seed, df_rows, model, train, test, valid, N, method=method, n_epochs=n_epochs, lr=lr)
                except Exception:
                    continue

        del x_train, y_train, x_test, y_test, x_valid, y_valid, train, test, valid
        gc.collect()

    df = pd.DataFrame(df_rows)
    df.to_csv("lr.csv")


if __name__ == "__main__":
    app.run(main)
