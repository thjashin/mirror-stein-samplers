library(MASS)
# library(selectiveInference)
library(devtools)
load_all('../../selectiveInference')
library(glmnet)


test_randomized = function(seed=1, outfile=NULL, type="selected", loss="ls", lambda_frac=0.7,
                           nrep=50, n=100, p=40, s=30, rho=0.3) {
  
  snr = sqrt(2*log(p)/n)

  set.seed(seed)
  # library(glue)

  dfs = list()
  j = 1
  for (target_coverage in c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)) {
    coverage = list(
      default=NULL,
      svgd=NULL,
      svmd=NULL
    )
    for (i in 1:nrep) {
      if (loss=="ls") {
        data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=snr, design="equicorrelated", scale=TRUE)
      } else if (loss=="logit") {
        data = selectiveInference:::logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
      }
      
      X=data$X
      y=data$y
      beta=data$beta
      cat("true nonzero:", which(beta!=0), "\n")

      sigma_est=1
      lambda = lambda_frac*selectiveInference:::theoretical.lambda(X, loss, sigma_est)  # theoretical lambda
      
      rand_lasso_soln = selectiveInference:::randomizedLasso(X, 
                                                            y, 
                                                            lambda*n, 
                                                            family=selectiveInference:::family_label(loss))
      if (is.null(rand_lasso_soln$active_set)) {
        next
      }
      active_vars=rand_lasso_soln$active_set
      cat("active_vars:",active_vars,"\n")

      targets=selectiveInference:::compute_target(rand_lasso_soln, type=type, sigma_est=sigma_est)
      
      print("default:")
      PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                    targets=targets,
                                                    sampler="norejection", #"norejection", #"adaptMCMC", #
                                                    level=target_coverage,
                                                    burnin=2000,
                                                    nsample=4000,
                                                    nchain=1)
      sel_coverage = selectiveInference:::compute_coverage(PVS$ci, beta[active_vars])
      coverage$default = c(coverage$default, sel_coverage)

      print("svgd:")
      PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                    targets=targets,
                                                    sampler="svgd", #"norejection", #"adaptMCMC", #
                                                    level=target_coverage,
                                                    burnin=0,
                                                    nsample=50,
                                                    nchain=40)
      sel_coverage = selectiveInference:::compute_coverage(PVS$ci, beta[active_vars])
      coverage$svgd = c(coverage$svgd, sel_coverage)

      print("svmd:")
      PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                    targets=targets,
                                                    sampler="svmd", #"norejection", #"adaptMCMC", #
                                                    level=target_coverage,
                                                    burnin=0,
                                                    nsample=50,
                                                    nchain=40)
      sel_coverage = selectiveInference:::compute_coverage(PVS$ci, beta[active_vars])
      coverage$svmd = c(coverage$svmd, sel_coverage)
      # sel_lengths=c(sel_lengths, as.vector(PVS$ci[,2]-PVS$ci[,1]))
    }
    df = data.frame(
      default=coverage$default,
      svgd=coverage$svgd,
      svmd=coverage$svmd,
      target=as.vector(rep(target_coverage, length(coverage$default)))
    )
    print(df)
    dfs[[j]] = df
    j = j + 1
  }

  print(dfs)
  df_total = do.call(rbind, dfs)
  write.csv(df_total, 'coverage.csv')
}

test_randomized(nrep=50, n=100, p=40, s=0, rho=0.3)
