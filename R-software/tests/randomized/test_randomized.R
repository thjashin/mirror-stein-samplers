library(MASS)
# library(selectiveInference)
library(devtools)
load_all('../../selectiveInference')
library(glmnet)


test_randomized = function(seed=1, outfile=NULL, type="selected", loss="ls", lambda_frac=0.7,
                           nrep=50, n=100, p=40, s=30, rho=0.3){
  
  snr = sqrt(2*log(p)/n)

  set.seed(seed)  
  construct_ci=TRUE
  penalty_factor = rep(1, p)
  
  pvalues = NULL
  blind_pvalues = NULL
  sel_intervals=NULL
  sel_coverages=NULL
  sel_lengths=NULL
  
  FDR_sample = NULL
  power_sample=NULL

  library(glue)
  # sampler = "norejection"
  # nsample = 10000
  sampler = "svgd"
  nsample = 200
  # sampler = "svmd"
  # nsample = 200
  pdf(glue("{sampler}.pdf"))

  for (i in 1:nrep){
    # set.seed(seed + i)
    
    if (loss=="ls"){
      # data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=snr)
      data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=snr, design="equicorrelated", scale=TRUE)
    } else if (loss=="logit"){
      data = selectiveInference:::logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
    }
    
    X=data$X
    y=data$y
    beta=data$beta
    cat("true nonzero:", which(beta!=0), "\n")
    
    #CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE, family=selectiveInference:::family_label(loss))
    #sigma_est=selectiveInference:::estimate_sigma(X,y,coef(CV, s="lambda.min")[-1]) # sigma via Reid et al.
    sigma_est=1
    #sigma_est = selectiveInference:::estimate_sigma_data_spliting(X,y)
    print(c("sigma est", sigma_est))
    
    # lambda = CV$lambda[which.min(CV$cvm+rnorm(length(CV$cvm))/sqrt(n))]  # lambda via randomized cv 
    lambda = lambda_frac*selectiveInference:::theoretical.lambda(X, loss, sigma_est)  # theoretical lambda
    # print(c("lambda:", lambda))
    
    rand_lasso_soln = selectiveInference:::randomizedLasso(X, 
                                                           y, 
                                                           lambda*n, 
                                                           family=selectiveInference:::family_label(loss))
    if (is.null(rand_lasso_soln$active_set)) {
      next
    }
    # print(summary(rand_lasso_soln$unpen_reg))
    glm_pvalues = coef(summary(rand_lasso_soln$unpen_reg))[,4]
    # print(glm_pvalues)

    targets=selectiveInference:::compute_target(rand_lasso_soln, type=type, sigma_est=sigma_est)
    
    PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                  targets=targets,
                                                  sampler=sampler, #"norejection", #"adaptMCMC", #
                                                  level=0.9,
                                                  burnin=1000, 
                                                  nsample=nsample,
                                                  nchain=10)
    # print("PVS:")
    # print(PVS)
    active_vars=rand_lasso_soln$active_set
    cat("active_vars:",active_vars,"\n")
    pvalues = c(pvalues, PVS$pvalues)
    blind_pvalues = c(blind_pvalues, glm_pvalues)
    sel_intervals = rbind(sel_intervals, PVS$ci)  # matrix with two rows
    
    
    if (length(pvalues)>0){
      print(pvalues)
      plot(ecdf(pvalues))
      plot(ecdf(blind_pvalues), add=TRUE, col='red')
      #lines(ecdf(naive_pvalues), col="red")
      abline(0,1)
    }
    
    if (construct_ci && length(active_vars)>0){
      print("ci:")
      print(PVS$ci)
      sel_coverages=c(sel_coverages, selectiveInference:::compute_coverage(PVS$ci, beta[active_vars]))
      sel_lengths=c(sel_lengths, as.vector(PVS$ci[,2]-PVS$ci[,1]))
      print(c("selective coverage:", mean(sel_coverages)))
      print(c("selective length mean:", mean(sel_lengths)))
      print(c("selective length median:", median(sel_lengths)))
      #naive_coverages=c(naive_coverages, selectiveInference:::compute_coverage(PVS$naive_intervals, beta[active_vars]))
      #naive_lengths=c(naive_lengths, as.vector(PVS$naive_intervals[2,]-PVS$naive_intervals[1,]))
      #print(c("naive coverage:", mean(naive_coverages)))
      #print(c("naive length mean:", mean(naive_lengths)))
      #print(c("naive length median:", median(naive_lengths)))
    }
    
    mc = selectiveInference:::selective.plus.BH(beta, active_vars, PVS$pvalues, q=0.2)
    FDR_sample=c(FDR_sample, mc$FDR)
    power_sample=c(power_sample, mc$power)
    
    if (length(FDR_sample)>0){
      print(c("FDR:", mean(FDR_sample)))
      print(c("power:", mean(power_sample)))
    }
  }
  
  if (is.null(outfile)){
    outfile=paste("randomized_", type, ".rds", sep="")
  }
  
  saveRDS(list(sel_intervals=sel_intervals, sel_coverages=sel_coverages, sel_lengths=sel_lengths,
               pvalues=pvalues,
               FDR_sample=FDR_sample, power_sample=power_sample,
               n=n,p=p, s=s, snr=snr, rho=rho, type=type), file=outfile)
  dev.off()
  return(list(pvalues=pvalues))
}

# test_randomized(n=100, p=40, s=4, rho=0.3)
test_randomized(n=100, p=40, s=0, rho=0.3)
