# -*- coding: utf-8 -*-

"""Compute various information criteria from MCMC chains.
If multiple chains are provided, compute the deviation.


Input: output['log_lik'], output['numparams'], data['nt'] 
Return:

WAIC: Watanabe-Akaike or widely applicable information criterion (prefreed to below criteria since it is fully Bayesian)
DIC: deviance information criterion (generalization of the AIC and the BIC)
AICC: (generalization of the AIC)
AIC: Akaike information criterion
BIC: Bayesian information criterion


k = nparams,
n = ndata,
s=nsamples,

log_lik: log of likelihood,  with size of s x n where s is number of samples and n is number of data points (Note: first sum over s and then over n)
maxlike: max of log_lik,


p_waic=sum of the variances of the log likelihoods across all observations i.e.,  
p_waic = np.sum(var(log_lik, axis=0, dtype=np.float64))

lpd= the sum of the log of the mean likelihoods across all observations i.e.,  
lpd=np.sum(np.log(np.mean(np.exp(log_lik),axis=0)))

elpd_waic = lpd - p_waic,
WAIC = -2 * elpd_waic

AIC =-2 log (maximum likelihood) + 2 (number of parameters)
AIC= 2 * k - 2 * log_lik.max()
AICc= AIC + 2 * k * (k + 1) / (n - k - 1)
BIC: -2 * log_lik.max() + k * log(n)

DIC=-2 * (Lat_param_mean-pDIC) 

Lat_param_mean= log likelihood of the data given the posterior means of the parameters 
pDIC  is twice the difference between Lat_param_mean, 
which can be thought of as a kind of best-fit log likelihood, 
and the average log likelihood across all samples from the posterior distribution




References
----------

Understanding predictive information criteria for Bayesian models: 
https://arxiv.org/pdf/1307.5928.pdf


Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Practical
Bayesian model evaluation using leave-one-out cross-validation
and WAIC. Statistics and Computing, 27(5):1413–1432.
doi:10.1007/s11222-016-9696-4. https://arxiv.org/abs/1507.04544


Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Pareto
smoothed importance sampling. https://arxiv.org/abs/arXiv:1507.02646v5



Resources:
https://github.com/JohannesBuchner/ic4stan
https://github.com/stan-dev/stan/issues/473
http://kylehardman.com/BlogPosts/View/7
"""


from __future__ import division # For Python 2 compatibility

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def maxlike(log_lik):
    return dict(maxlike=np.max(log_lik))


def waic(log_lik):
    lpd=np.sum(np.log(np.mean(np.exp(log_lik), axis=0)))
    p_waic = np.sum(np.var(log_lik, axis=0, dtype=np.float64))
    elpd_waic=lpd - p_waic
    waic = -2 * elpd_waic
    return dict(p_waic=p_waic, elpd_waic=elpd_waic, waic=waic)


def lik_at_mean(log_lik, uparams):
    # compute diagonal Mahalanobis distance to pick the closest point
    distances = []
    for row in zip(*uparams):
        # flatten parameters
        row_params_flat = np.concatenate([y.flatten() for y in row])
        # compute distance to 0
        distances.append(np.linalg.norm(row_params_flat))
    # pick the one with the smallest distance
    i = np.argmin(distances)
    #print ('mean parameters at %d of %d' % (i, len(distances)))
    #print ('maximum likelihood at %d' % np.argmax(log_lik))
    
    return log_lik[i]


def dic(log_lik, uparams):
    pDIC = 2 * (lik_at_mean(log_lik, uparams) - np.mean(log_lik, axis=0))
    pDIC_alt = 2*(np.var(log_lik, axis=0, dtype=np.float64))
    DIC = -2 * (lik_at_mean(log_lik, uparams)-pDIC) 
    return DIC


def aic(log_lik, nparams):
    k = nparams
    return 2 * k - 2 * np.max(log_lik)


def aicc(log_lik, nparams, ndata):
    k = nparams
    n = ndata
    return aic(log_lik, nparams) + 2 * k * (k + 1) / (n - k - 1)


def bic(log_lik, nparams, ndata):
    k = nparams
    n = ndata
    return -2 * np.max(log_lik) + k * np.log(n)


# if __name__ == '__main__':
#
#     import os
#     import sys
#     from parse_csv import parse_csv
#
#     # input_filenames = sys.argv[1:2]
#     # for filename in input_filenames:
#     #     data = np.load(filename)
#     #     ndata =data['nt']
#
#     results = {}
#     output_filenames = sys.argv[3:]
#     print('all the output files: ', output_filenames)
#
#     for filename in output_filenames:
#         print('for filename', filename)
#         #output = np.load(filename)
#         output = parse_csv(filename)
#         likelihood = -1*output['log_lik']
#
#         uparams=[]
#         for k, v in output.items():
#             uparams.append((v - np.mean(v, axis=0)) / np.std(v, axis=0))
#
#         nparams=int(output['num_params'][0])
#         print (  'num of parameters:' , nparams)
#         ndata =int(output['num_data'][0])
#         print (  'num of data:' , ndata)
#         nsamples = np.shape(likelihood)[0]
#         print (  'num of samples:' , nsamples)
#
#         result = maxlike(likelihood)
#         result.update(waic(likelihood))
#         result['aicc'] = aicc(likelihood, nparams, ndata)
#         result['aic'] = aic(likelihood, nparams)
#         result['dic'] = dic(likelihood, uparams)
#         result['bic'] = bic(likelihood, nparams, ndata)
#
#         for k, v in result.items():
#             results[k] = results.get(k, []) + [v]
#             print ('%-20s | %-10s | %.1f' % (filename, k, np.mean(v)))
#         print()
#     print ('%-10s : mean\tstd' % 'criterion')
#
#     for k, v in results.items():
#         print ('%-10s : %.1f\t%.1f' % (k, np.mean(v), np.std(v)))
#
#
#     with open('report_IC_'+sys.argv[2]+'.txt', 'w') as fd:
#         fmt = '%-10s , %.1f\t, %.1f\n'
#         for k, v in results.items():
#             args = (k, np.mean(v), np.std(v))
#             fd.write(fmt % args)
#
# print("Writing IC completed")


# In LOO.R package: 
  # lppd <- sum (log (colMeans(exp(log_lik))))
  # p_waic_1 <- 2*sum (log(colMeans(exp(log_lik))) - colMeans(log_lik))
  # p_waic_2 <- sum (colVars(log_lik))