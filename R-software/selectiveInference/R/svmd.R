# library(tensorflow)
# # library(tfautograph)


# cross_diff = tf_function(function(x1, x2) {
#   x1 = x1[all_dims(), , tf$newaxis, ]
#   x2 = x2[all_dims(), tf$newaxis, , ]
#   return(x1 - x2)
# })


# compute_squared_dist = tf_function(function(cross_diff_val) {
#   return(tf$reduce_sum(tf$square(cross_diff_val), axis=-1L))
# })


# heuristic_kernel_width = tf_function(function(squared_dists) {
#   # square_dists: [..., K1, K2]
#   # return: [...]
#   sh = squared_dists$shape$as_list()
#   n_elems = sh[[length(sh)]] * sh[[length(sh) - 1]]
#   top_k_values = tf$nn$top_k(
#     tf$reshape(squared_dists, shape(-1, n_elems)), k=n_elems %/% 2L, sorted=FALSE)$values
#   kernel_width2 = tf$reduce_min(top_k_values, axis=-1L)
#   kernel_width2 = tf$where(
#     tf$equal(kernel_width2, 0), tf$ones_like(kernel_width2), kernel_width2)
#   if (length(sh) > 2) {
#     kernel_width2 = tf$reshape(kernel_width2, squared_dists$shape[1:(length(sh) - 2)])
#   } else {
#     kernel_width2 = tf$reshape(kernel_width2, shape())
#   }
#   return(tf$stop_gradient(kernel_width2))
# })


# rbf = tf_function(function(x1, x2, kernel_width2=NULL, left_grad=FALSE) {
#   # x1: [..., K, D]
#   # x2: [..., K, D]
#   ret = list(
#     gram=NULL,
#     kernel_width2=NULL,
#     grad_x1=NULL
#   )
#   cross_x1_x2 = cross_diff(x1, x2)
#   squared_dists = compute_squared_dist(cross_x1_x2)
#   if (is.null(kernel_width2)) {
#     kernel_width2 = heuristic_kernel_width(squared_dists)
#   }
#   # gram: [..., K, K]
#   ret$gram = tf$exp(-squared_dists / kernel_width2[all_dims(), tf$newaxis, tf$newaxis])
#   ret$kernel_width2 = kernel_width2
#   if (left_grad) {
#     # grad_x1: [..., K, K, D]
#     ret$grad_x1 = -2 * cross_x1_x2 / kernel_width2[all_dims(), tf$newaxis, tf$newaxis, tf$newaxis] * ret$gram[all_dims(), tf$newaxis]
#   }
#   if (length(ret) == 1L) {
#     return(ret[[1]])
#   }
#   return(ret)
# })


# imq = tf_function(function(x1, x2, kernel_width2=NULL, left_grad=FALSE) {
#   # x1: [..., K, D]
#   # x2: [..., K, D]
#   ret = list(
#     gram=NULL,
#     kernel_width2=NULL,
#     grad_x1=NULL
#   )
#   cross_x1_x2 = cross_diff(x1, x2)
#   squared_dists = compute_squared_dist(cross_x1_x2)
#   if (is.null(kernel_width2)) {
#     kernel_width2 = heuristic_kernel_width(squared_dists)
#   }
#   inner = tf$constant(1., dtype=tf$float64) + squared_dists / kernel_width2[all_dims(), tf$newaxis, tf$newaxis]
#   ret$gram = tf$constant(1., dtype=tf$float64) / tf$sqrt(inner)
#   ret$kernel_width2 = kernel_width2
#   if (left_grad) {
#     # grad_x1: [..., K, K, D]
#     ret$grad_x1 = -(inner^(-1.5))[all_dims(), tf$newaxis] * cross_x1_x2 / kernel_width2[all_dims(), tf$newaxis, tf$newaxis, tf$newaxis]
#   }
#   if (length(ret) == 1L) {
#     return(ret[[1]])
#   }
#   return(ret)
# })


# nystrom = tf_function(function(Kxz, eigval, eigvec, grad_Kxz=NULL) {
#   # Kxz: [..., N, M], grad_Kxz: [..., N, M, D]
#   # eigvec: [..., M, n_eigen]
#   # eigval: [..., n_eigen]
#   Kxz_shape = Kxz$shape$as_list()
#   M = tf$cast(Kxz_shape[[length(Kxz_shape)]], tf$float64)
#   # # z: [..., M, D]
#   # # x: [..., N, D]
#   # if (grad) {
#   #   kern_list = kernel(x, z, kernel_width2=kernel_width2, left_grad=TRUE)
#   #   Kxz = kern_list$gram
#   #   grad_Kxz = kern_list$grad_x1
#   # } else {
#   #   Kxz = kernel(x, z, kernel_width2=kernel_width2)
#   # }

#   # ret: [..., N, n_eigen]
#   u = tf$sqrt(M) * tf$matmul(Kxz, eigvec) / eigval[all_dims(), tf$newaxis, ]
#   if (!is.null(grad_Kxz)) {
#     # grad_u: [N, n_eigen, D]
#     grad_u = tf$sqrt(M) * tf$einsum("...nml,...mj->...njl", grad_Kxz, eigvec) / eigval[all_dims(), tf$newaxis, , tf$newaxis]
#     return(list(u=u, grad_u=grad_u))
#   } else {
#     return(u)
#   }
# })


# compute_K_psi = function(mirror_map, theta, kernel, n_eigen_threshold, md=FALSE, jitter=1e-5) {
#   K = theta$shape[[1]]
#   theta_ = tf$stop_gradient(theta)
#   # gram: [K, K]
#   kern_list = kernel(theta_, theta_)
#   gram = kern_list$gram
#   kernel_width2 = kern_list$kernel_width2
#   gram = gram + tf$constant(jitter, dtype=tf$float64) * tf$eye(K, dtype=tf$float64)
#   # eigval: [K]
#   # eigvec: [K, K]
#   eig_list = tf$linalg$eigh(gram)
#   eigval = eig_list[[1]]
#   eigvec = eig_list[[2]]

#   if (!is.null(n_eigen_threshold)) {
#     eigen_arr = tf$reverse(eigval, axis=list(-1L))
#     eigen_arr = eigen_arr / tf$reduce_sum(eigen_arr)
#     eigen_cum = tf$cumsum(eigen_arr, axis=-1L)
#     n_eigen = tf$reduce_sum(
#       tf$cast(tf$less(eigen_cum, n_eigen_threshold), tf$int32)) + 1L
#     # eigval: [n_eigen]
#     # eigvec: [K, n_eigen]
#     eigval = eigval[-n_eigen:NULL]
#     eigvec = eigvec[ , -n_eigen:NULL]
#   }

#   # mu: [n_eigen]
#   mu = eigval / K
#   # v: [K, n_eigen]
#   v = eigvec * tf$sqrt(tf$cast(K, tf$float64))
#   # use nystrom formula to fix the gradient
#   # v_theta: [K, n_eigen]
#   # v_theta_grad: [K, n_eigen, D - 1]
#   nystrom_list = nystrom(
#       theta_, theta, eigval, eigvec, kernel_width2=kernel_width2, kernel=kernel, grad=TRUE)
#   v_theta = nystrom_list$u
#   v_theta_grad = nystrom_list$grad_u

#   if (md) {
#     with(tf$GradientTape() %as% tape, {
#       tape$watch(theta)
#       # nabla2_psi_theta: [K, D - 1, D - 1]
#       nabla2_psi_theta = mirror_map$nabla2_psi(theta)
#     })
#     # grad_nabla2_psi: [K, D - 1, D - 1, D - 1]
#     grad_nabla2_psi = tape$batch_jacobian(nabla2_psi_theta, theta)
#     # Gamma: [n_eigen, n_eigen, D - 1, D - 1]
#     Gamma = tf$constant(1 / K, dtype=tf$float64) * tf$einsum("ki,kj,kab->ijab", v_theta, v_theta, nabla2_psi_theta)
#   } else {
#     # nabla2_psi_theta: [K, D - 1, D - 1]
#     nabla2_psi_theta = mirror_map$nabla2_psi(theta_)
#     # Gamma: [n_eigen, n_eigen, D - 1, D - 1]
#     Gamma = tf$constant(1 / K, dtype=tf$float64) * tf$einsum("ki,kj,kab->ijab", v, v, nabla2_psi_theta)
#   }

#   mu_sqrt = tf$sqrt(mu)
#   # K_psi: [K, K, D - 1, D - 1]
#   K_psi = tf$einsum("i,j,ki,lj,ijab->klab", mu_sqrt, mu_sqrt, v, v_theta, Gamma)
#   # nabla2_psi_theta_inv: [K, D - 1, D - 1]
#   nabla2_psi_theta_inv = mirror_map$nabla2_psi_inv(theta)
#   # repul_grad_term1: [K, n_eigen, n_eigen, D - 1]
#   repul_grad_term1 = tf$einsum("ljd,ijab,lbd->lija", v_theta_grad, Gamma, nabla2_psi_theta_inv)
#   # grad_nabla2_psi_inv: [K, D - 1, D - 1]
#   grad_nabla2_psi_theta_inv_diag = mirror_map$grad_nabla2_psi_inv_diag(theta)
#   # repul_grad_term2: [K, n_eigen, n_eigen, D - 1]
#   repul_grad_term2 = tf$einsum("lj,ijab,lbd->lija", v_theta, Gamma, grad_nabla2_psi_theta_inv_diag)
#   repul_grad = repul_grad_term1 + repul_grad_term2

#   if (md) {
#     # TODO: can be simplified since the last two terms will reduce to identity
#     term31 = tf$einsum("lj,lid,lj,lab,lbd->lija", 
#                         v_theta, v_theta_grad, v_theta, nabla2_psi_theta, nabla2_psi_theta_inv)
#     term32 = tf$einsum("lj,li,ljd,lab,lbd->lija",
#                         v_theta, v_theta, v_theta_grad, nabla2_psi_theta, nabla2_psi_theta_inv)
#     term33 = tf$einsum("lj,li,lj,labd,lbd->lija",
#                         v_theta, v_theta, v_theta, grad_nabla2_psi, nabla2_psi_theta_inv)
#     repul_grad_term3 = tf$constant(1 / K, dtype=tf$float64) * (term31 + term32 + term33)
#     repul_grad = repul_grad + repul_grad_term3
#   }

#   # repulsive_term: [K, K, D - 1]
#   repulsive_term = tf$einsum("i,j,ki,lija->kla", mu_sqrt, mu_sqrt, v, repul_grad)
#   # repulsive_term: [K, D - 1]
#   repulsive_term = tf$reduce_mean(repulsive_term, axis=1L)
#   return(list(
#     K_psi=K_psi, 
#     repulsive_term=repulsive_term)
#   )
# }


# # IMPORTANT: need to be moved into the scope of mirror sampler before using
# svmd_update = tf_function(function(theta, n_eigen_threshold=0.98, md=FALSE) {
# # svmd_update = function(target, theta, kernel, n_eigen_threshold=0.98, md=FALSE) {
#   K = theta$shape[[1]]
#   K_psi_list = compute_K_psi(target$mirror_map, theta, imq, n_eigen_threshold, md=md)
#   K_psi = K_psi_list$K_psi
#   repulsive_term = K_psi_list$repulsive_term
#   # nabla2_psi_theta_inv_grad: [K, D]
#   nabla2_psi_theta_inv_grad = target$nabla_psi_inv_grad_logp(theta)
#   # weighted_grad: [K, D]
#   weighted_grad = tf$constant(1 / K, dtype=tf$float64) * tf$einsum("klab,lb->ka", K_psi, nabla2_psi_theta_inv_grad)
#   return(weighted_grad + repulsive_term)
# })


# truncate_and_grad = tf_function(function(eigval, eigvec, n_eigen_threshold, Kxz, grad_Kxz) {
#   eigval_dims = length(eigval$shape$as_list())
#   K = eigval$shape[[eigval_dims - 1]]
#   eigen_arr = tf$reduce_mean(tf$reshape(eigval, shape(-1, K)), axis=0L)
#   eigen_arr = tf$reverse(eigen_arr, axis=list(-1L))
#   eigen_arr = eigen_arr / tf$reduce_sum(eigen_arr)
#   eigen_cum = tf$cumsum(eigen_arr, axis=-1L)
#   n_eigen = tf$reduce_sum(
#     tf$cast(tf$less(eigen_cum, n_eigen_threshold), tf$int32)) + 1L
#   # eigval: [..., n_eigen]
#   # eigvec: [..., K, n_eigen]
#   # n_eigen is tensor, thus following python convention of indices
#   eigval = eigval[all_dims(), -n_eigen:NULL]
#   eigvec = eigvec[all_dims(), -n_eigen:NULL]
#   # mu: [..., n_eigen]
#   mu = eigval / K
#   # v: [..., K, n_eigen]
#   v = eigvec * tf$sqrt(tf$cast(K, tf$float64))
#   # use nystrom formula to fix the gradient
#   # v_theta: [..., K, n_eigen]
#   # v_theta_grad: [..., K, n_eigen, D]
#   nystrom_list = nystrom(Kxz, eigval, eigvec, grad_Kxz=grad_Kxz)
#   v_theta = nystrom_list$u
#   v_theta_grad = nystrom_list$grad_u
#   return(list(mu=mu, v=v, v_theta=v_theta, v_theta_grad=v_theta_grad))
# })


# eigen_quantities = function(theta_, theta, kernel, n_eigen_threshold, jitter=1e-5) {
#   theta_dims = length(theta$shape$as_list())
#   K = theta$shape[[theta_dims - 1]]
#   # gram: [..., K, K]
#   kern_list = kernel(theta, theta_, left_grad=TRUE)
#   Kxz = kern_list$gram
#   grad_Kxz = kern_list$grad_x1
#   gram = tf$stop_gradient(Kxz) + tf$constant(jitter, dtype=tf$float64) * tf$eye(K, dtype=tf$float64)
#   # eigval: [..., K]
#   # eigvec: [..., K, K]
#   with(tf$device("/cpu:0"), {
#     eig_list = tf$linalg$eigh(gram)
#   })
#   eigval = eig_list[[1]]
#   eigvec = eig_list[[2]]
#   return(truncate_and_grad(eigval, eigvec, n_eigen_threshold, Kxz, grad_Kxz))
# }


# mirror_sampler = function(target,
#                           theta_init,
#                           nsamples,
#                           nchain=1,
#                           method="svmd",
#                           kernel=imq,
#                           learning_rate=0.005,
#                           n_iters=3000) {

#   svmd_grad = tf_function(function(theta, theta_, mu, v, v_theta, v_theta_grad) {
#     theta_dims = length(theta$shape$as_list())
#     K = theta$shape[[theta_dims - 1]]

#     # nabla2_psi_theta: [..., K, D, D]
#     nabla2_psi_theta = target$mirror_map$nabla2_psi(theta_)
#     # nabla2_psi_theta_inv: [..., K, D, D]
#     nabla2_psi_theta_inv = target$mirror_map$nabla2_psi_inv(theta)
#     # nabla2_psi_theta_inv_grad: [..., K, D]
#     nabla2_psi_theta_inv_grad = target$nabla_psi_inv_grad_logp(theta)
#     # grad_nabla2_psi_inv: [..., K, D, D]
#     grad_nabla2_psi_theta_inv_diag = target$mirror_map$grad_nabla2_psi_inv_diag(theta)

#     mu_sqrt = tf$sqrt(mu)
#     # [..., K, K]
#     i_reduced = tf$einsum("...i,...ki,...mi->...km", mu_sqrt, v, v)
#     coeff = tf$constant(1 / (K * K), dtype=tf$float64)
#     # weighted_grad: [..., K, D]
#     weighted_grad = coeff * tf$einsum(
#         "...km,...j,...lj,...mj,...mab,...lb->...ka", i_reduced, mu_sqrt, v_theta, v, nabla2_psi_theta, nabla2_psi_theta_inv_grad)

#     # repul_term1: [..., K, D]
#     repul_term1 = coeff * tf$einsum(
#         "...km,...j,...ljd,...mj,...mab,...lbd->...ka", i_reduced, mu_sqrt, v_theta_grad, v, nabla2_psi_theta, nabla2_psi_theta_inv)
#     # repul_term2: [..., K, D]
#     repul_term2 = coeff * tf$einsum(
#         "...km,...j,...lj,...mj,...mab,...lbd->...ka", i_reduced, mu_sqrt, v_theta, v, nabla2_psi_theta, grad_nabla2_psi_theta_inv_diag)
#     # repulsive_term: [..., K, D]
#     repulsive_term = repul_term1 + repul_term2

#     return(weighted_grad + repulsive_term)
#   })

#   svmd_update_v2 = function(theta, n_eigen_threshold=0.98) {
#   # svmd_update_v2 = function(theta, n_eigen_threshold=0.98) {
#     theta_ = tf$stop_gradient(theta)
#     # mu: [..., n_eigen]
#     # v, v_theta: [..., K, n_eigen]
#     # v_theta_grad: [..., K, n_eigen, D]
#     eigen_list = eigen_quantities(theta_, theta, kernel, n_eigen_threshold)
#     mu = eigen_list$mu
#     v = eigen_list$v
#     v_theta = eigen_list$v_theta
#     v_theta_grad = eigen_list$v_theta_grad

#     return(svmd_grad(theta, theta_, mu, v, v_theta, v_theta_grad))
#   }

#   svgd_update = tf_function(function(eta, theta) {
#   # svgd_update = function(eta, theta) {
#     n_dims = length(theta$shape$as_list())
#     K = theta$shape[[n_dims - 1]]
#     # grad_logp: [..., K, D]
#     grad_logp_eta = target$dual_grad_logp(eta, theta=theta)
#     # gram: [..., K, K], grad_gram: [..., K, K, D]
#     kern_list = kernel(theta, theta, left_grad=TRUE)
#     gram = kern_list$gram
#     grad_gram = kern_list$grad_x1
#     # nabla2_psi_inv: [..., K, D, D]
#     nabla2_psi_theta_inv = target$mirror_map$nabla2_psi_inv(theta)
#     # repulsive_term: [..., K, D]
#     repulsive_term = tf$einsum("...iab,...ijb->...ja", nabla2_psi_theta_inv, grad_gram) / K
#     # weighted_grad: [..., K, D]
#     weighted_grad = tf$matmul(gram, grad_logp_eta) / K
#     return(weighted_grad + repulsive_term)
#   })

#   # svmd_wrapper = function(x) {
#   #   return(svmd_update_v2(x, n_eigen_threshold=0.98))
#   # }

#   # multichain_svmd = tf_function(function(theta) {
#   #   return(tf$map_fn(svmd_wrapper, theta, parallel_iterations=10))
#   # })

#   # svgd_wrapper = function(x) {
#   #   return(svgd_update(x[[1]], x[[2]]))
#   # }

#   # multichain_svgd = tf_function(function(eta, theta) {
#   #   return(tf$map_fn(svgd_wrapper, list(eta, theta), fn_output_signature=eta$dtype, parallel_iterations=10))
#   # })

#   K = nsamples
#   D = theta_init$shape[[1]]
#   # g = tf$random$Generator$from_seed(1L)
#   eta0 = target$mirror_map$nabla_psi(theta_init[tf$newaxis,]) + tf$random$normal(shape(nchain, K, D), dtype=tf$float64)
#   # eta: [nchain, K, D]
#   eta = tf$Variable(eta0)
#   theta = target$mirror_map$nabla_psi_star(eta)
#   optimizer = tf$keras$optimizers$RMSprop(learning_rate)
#   pb = txtProgressBar(min=0, max=n_iters, initial=0) 
#   for (t in 1:n_iters) {
#     if (method == "svmd") {
#       # for 2D problem, 0.998 works well?
#       eta_grad = svmd_update_v2(theta, n_eigen_threshold=0.98)
#       # eta_grad = multichain_svmd(theta)
#     } else if (method == "svgd") {
#       eta_grad = svgd_update(eta, theta)
#       # eta_grad = multichain_svgd(eta, theta)
#     } else {
#       stop("Not implemented.")
#     }
#     optimizer$apply_gradients(list(list(-eta_grad, eta)))
#     theta = target$mirror_map$nabla_psi_star(eta)
#     setTxtProgressBar(pb, t)
#   }
#   return(tf$reshape(theta, shape(-1, D)))
# }


# safe_log = function(x) {
#   return(tf$math$log(tf$maximum(tf$constant(1e-32, dtype=x$dtype), x)))
#   # return(tf$math$log(x))
# }


# safe_reciprocal = function(x) {
#   return(tf$constant(1., dtype=tf$float64) / tf$maximum(x, tf$constant(1e-32, dtype=x$dtype)))
#   # return(1. / x)
# }


# nonnegative_entropic_map = function() {
#   psi_star = function(eta) {
#     # ret: [..., K]
#     return(tf$reduce_sum(tf$exp(eta), axis=-1L))
#   }

#   nabla_psi = function(theta) {
#     # ret: [..., K, D]
#     return(safe_log(theta))
#   }

#   nabla_psi_star = function(eta) {
#     # ret: [..., K, D]
#     return(tf$exp(eta))
#   }

#   nabla2_psi = function(theta) {
#     # theta: [..., K, D]
#     # ret: [..., K, D, D]
#     return(tf$linalg$diag(safe_reciprocal(theta)))
#   }

#   nabla2_psi_inv = function(theta) {
#     # theta: [..., K, D]
#     # ret: [..., K, D, D]
#     return(tf$linalg$diag(theta))
#   }

#   nabla2_psi_inv_mul = function(theta, rhs) {
#     # rhs: [..., K, D, ?]
#     return(theta[all_dims(), tf$newaxis] * rhs)
#   }

#   logdet_nabla2_psi_star = function(eta) {
#     # eta: [..., K, D]
#     # ret: [..., K]
#     return(tf$reduce_sum(eta, axis=-1L))
#   }

#   grad_logdet_nabla2_psi_star = function(eta, theta=NULL) {
#     # eta: [..., K, D]
#     # ret: [..., K, D]
#     return(tf$ones_like(eta))
#   }

#   grad_nabla2_psi_inv_diag = function(theta) {
#     # ret: [..., K, D, D]
#     n_dims = length(theta$shape$as_list())
#     return(tf$eye(theta$shape[[n_dims]], batch_shape=theta$shape[1:(n_dims - 1)], dtype=tf$float64))
#   }

#   return(list(
#     psi_star=psi_star,
#     nabla_psi=nabla_psi,
#     nabla_psi_star=nabla_psi_star,
#     nabla2_psi=nabla2_psi,
#     nabla2_psi_inv=nabla2_psi_inv,
#     nabla2_psi_inv_mul=nabla2_psi_inv_mul,
#     logdet_nabla2_psi_star=logdet_nabla2_psi_star,
#     grad_logdet_nabla2_psi_star=grad_logdet_nabla2_psi_star,
#     grad_nabla2_psi_inv_diag=grad_nabla2_psi_inv_diag
#   ))
# }


# build_target = function(logp,
#                         mirror_map,
#                         grad_logp=NULL,
#                         nabla_psi_inv_grad_logp=NULL,
#                         dual_logp=NULL,
#                         dual_grad_logp=NULL) {
#   if (is.null(grad_logp)) {
#     grad_logp = function(theta) {
#       # ret: [..., K, D]
#       with(tf$GradientTape() %as% tape, {
#         tape$watch(theta)
#         logp_theta = tf$reduce_sum(logp(theta))
#       })
#       return(tape$gradient(logp_theta, theta))
#     }
#   }

#   if (is.null(nabla_psi_inv_grad_logp)) {
#     nabla_psi_inv_grad_logp = function(theta) {
#       # ret: [..., K, D]
#       grad_logp_theta = grad_logp(theta)
#       # return(tf$squeeze(tf$matmul(mirror_map$nabla2_psi_inv(theta), grad_logp_theta[ , , tf$newaxis]), axis=-1L))
#       return(tf$squeeze(mirror_map$nabla2_psi_inv_mul(theta, grad_logp_theta[all_dims(), tf$newaxis]), axis=-1L))
#     }
#   }

#   if (is.null(dual_logp)) {
#     dual_logp = function(eta, theta=NULL) {
#       # ret: [..., K]
#       if (is.null(theta)) {
#         theta = mirror_map$nabla_psi_star(eta)
#       }
#       return(logp(theta) + mirror_map$logdet_nabla2_psi_star(eta))
#     }
#   }

#   if (is.null(dual_grad_logp)) {
#     dual_grad_logp = function(eta, theta=NULL) {
#       # ret: [..., K, D]
#       if (is.null(theta)) {
#         theta = mirror_map$nabla_psi_star(eta)
#       }
#       return(nabla_psi_inv_grad_logp(theta) + mirror_map$grad_logdet_nabla2_psi_star(eta, theta=theta))
#     }
#   }

#   return(list(
#     logp=logp,
#     mirror_map=mirror_map,
#     grad_logp=grad_logp,
#     nabla_psi_inv_grad_logp=nabla_psi_inv_grad_logp,
#     dual_logp=dual_logp,
#     dual_grad_logp=dual_grad_logp
#   ))
# }


# nonnegative_sampler = function(noise_scale, 
#                                observed, 
#                                linear_term, 
#                                offset_term,
#                                nsamples=200,
#                                nchain=1,
#                                method="svmd",
#                                learning_rate=0.01,
#                                n_iters=3000) {
#   noise_scale = tf$constant(noise_scale, dtype=tf$float64)
#   # linear_term: [|E|, |E|]
#   linear_term = tf$constant(linear_term, dtype=tf$float64)
#   # offset_term: [|E|]
#   offset_term = tf$constant(as.vector(offset_term), dtype=tf$float64)
#   # theta_init: [|E|]
#   theta_init = tf$constant(as.vector(observed), dtype=tf$float64)
#   if (length(observed) == 1) {
#     theta_init = theta_init[tf$newaxis]
#   }

#   logp = function(theta) {
#     # theta: [..., K, |E|]
#     # recon: [..., K, |E|]
#     recon = tf$einsum("...ki,ji->...kj", theta, linear_term) + offset_term
#     # ret: [..., K]
#     return(-tf$reduce_sum(tf$square(recon), axis=-1L) / (2 * tf$square(noise_scale)))
#   }

#   grad_logp = function(theta) {
#     # recon: [..., K, |E|]
#     # recon = tf$matmul(theta, linear_term, transpose_b=TRUE) + offset_term
#     recon = tf$einsum("...ki,ji->...kj", theta, linear_term) + offset_term
#     # ret: [..., K, |E|]
#     ret = -tf$einsum("...kj,ji->...ki", recon, linear_term) / tf$square(noise_scale)
#     return(ret)
#   }

#   mirror_map = nonnegative_entropic_map()
#   target = build_target(logp, mirror_map, grad_logp=grad_logp)

#   # samples: [K, |E|]
#   time = Sys.time()
#   samples = mirror_sampler(target,
#                            theta_init,
#                            nsamples,
#                            nchain=nchain,
#                            method=method,
#                            kernel=imq,
#                            learning_rate=learning_rate,
#                            n_iters=n_iters)
#   print(c("time:", difftime(Sys.time(), time, units="secs")))

#   return(as.matrix(samples))
# }
