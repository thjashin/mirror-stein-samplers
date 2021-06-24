import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
# os.environ['R_HOME'] = "/usr/lib/R"
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from rpy2 import rinterface
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from tqdm import tqdm

from selective.target import SelectiveTarget
from selective.data import HIV_NRTI
from constrained.nonnegative.entropic import NonnegativeEntropicMap
from constrained.sampling.kernel import imq
from constrained.sampling.svgd import svgd_update
from constrained.sampling.svmd import svmd_update_v2


devtools = importr("devtools")
r("load_all('R-software/selectiveInference')")
glmnet = importr("glmnet")
r("set.seed(1)")


def run(target, theta_init, method="svmd", K=50, n_chain=100):
    D = theta_init.shape[-1]
    eta0 = target.mirror_map.nabla_psi(theta_init[None, :]) + tf.random.normal([n_chain, K, D], dtype=tf.float64)
    # eta: [n_chain, K, D - 1]
    eta = tf.Variable(eta0)
    theta = target.mirror_map.nabla_psi_star(eta)
    n_iters = 1000
    kernel = imq
    trange = tqdm(range(n_iters))
    optimizer = tf.keras.optimizers.RMSprop(0.01)
#     optimizer = tf.keras.optimizersSGD(lr)
    for t in trange:
        if method == "svmd":
            eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.98)
        elif method == "svgd":
            eta_grad = svgd_update(target, eta, theta, kernel)
        else:
            raise NotImplementedError()
        optimizer.apply_gradients([(-eta_grad, eta)])
        theta = target.mirror_map.nabla_psi_star(eta)
    return tf.reshape(theta, [-1, theta.shape[-1]]).numpy()


data = "hiv"

if data == "hiv":
    X, _, _ = HIV_NRTI()
    n, p = X.shape
    s = 10
    sigma = 1
    r.assign("n", n)
    r.assign("p", p)
    r.assign("X", X)
    r.assign("s", s)
    r("""
    rho = 0.3
    lambda_frac = 0.7
    """)
else:
    r("""
    n = 100
    p = 40
    s = 0
    rho = 0.3
    lambda_frac = 0.7
    """)

nrep = 200
rng = np.random.RandomState(1)
seeds = rng.randint(1, 1e6, size=nrep)
nonneg_map = NonnegativeEntropicMap()


df = pd.DataFrame(columns=["target", "method", "covered", "width"])
df_rows = []
methods = ["default", "svgd", "svmd"]
# methods = ["default"]
target_coverages = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
for i in range(nrep):
    r.assign("seed", seeds[i])
    r("set.seed(seed)")
    np.random.seed(seeds[i])
    tf.random.set_seed(seeds[i])
    print("RUN {}".format(i))

    if data == "hiv":
        truth = np.zeros(p)
        truth[:s] = np.linspace(0.5, 1, s)
        np.random.shuffle(truth)
        truth /= np.sqrt(n)
        truth *= sigma
        y = X.dot(truth) + sigma * np.random.standard_normal(n)
        r.assign("y", y)
        r.assign("beta", truth)
    else:
        r("data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=sqrt(2*log(p)/n), design='equicorrelated', scale=TRUE)")

        r("X = data$X")
        r("y = data$y")
        r("beta = data$beta")
        r("cat('true nonzero:', which(beta != 0), '\n')")

    r("sigma_est = 1")
    # theoretical lambda
    r("lambda = lambda_frac*selectiveInference:::theoretical.lambda(X, 'ls', sigma_est)")

    r("rand_lasso_soln = selectiveInference:::randomizedLasso(X, y, lambda*n, family='gaussian')")
    rand_lasso_soln = r["rand_lasso_soln"]
    active_vars = rand_lasso_soln.rx2["active_set"]
    if active_vars is rinterface.NULL:
        continue
    print("active_vars:", active_vars)

    r("targets = selectiveInference:::compute_target(rand_lasso_soln, type='selected', sigma_est=sigma_est)")
    r("target_samples = mvrnorm(5000, rep(0,length(rand_lasso_soln$active_set)), targets$targets$cov_target)")

    r("linear = rand_lasso_soln$law$sampling_transform$linear_term")
    r("offset = rand_lasso_soln$law$sampling_transform$offset_term")
    r("theta_init = rand_lasso_soln$law$observed_opt_state")
    r("noise_scale = rand_lasso_soln$noise_scale")

    A = np.asarray(r["linear"])
    b = np.squeeze(np.asarray(r["offset"]), -1)
    theta_init = np.asarray(r["theta_init"])
    target = SelectiveTarget(nonneg_map, A, b, np.asarray(r["noise_scale"])[0])

    for method in methods:
        print("sampler: {}".format(method))
        if method == "default":
            r("opt_samples = get_opt_samples(rand_lasso_soln, sampler='norejection', nsample=7000, burnin=2000)")
        else:
            timer = -time.time()
            opt_samples = run(target, theta_init, method=method, K=50, n_chain=100)
            print("time:", timer + time.time())
            nr, nc = opt_samples.shape
            opt_samples_r = r.matrix(opt_samples, nrow=nr, ncol=nc)
            r.assign("opt_samples", opt_samples_r)

        for target_coverage in target_coverages:
            r.assign("target_coverage", target_coverage)
            r("""
            PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                          targets=targets,
                                                          level=target_coverage,
                                                          opt_samples=opt_samples,
                                                          target_samples=target_samples)
            """)
            r("sel_coverage = selectiveInference:::compute_coverage(PVS$ci, beta[rand_lasso_soln$active_set])")
            r("sel_length = as.vector(PVS$ci[,2] - PVS$ci[,1])")
            for covered, width in zip(r["sel_coverage"], r["sel_length"]):
                df_rows.append({"target": target_coverage,
                                "method": method,
                                "covered": covered,
                                "width": width})


df = pd.DataFrame(df_rows)
if data == "hiv":
    df.to_csv('hiv_cov.csv')
else:
    df.to_csv("coverage.csv")
