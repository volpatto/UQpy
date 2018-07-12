from library import *
import scipy.stats as stats
import numpy as np
import os
import sys
import copy
from scipy.spatial.distance import pdist
from modelist import *
import emcee
import corner
from scipy import integrate
from SampleMethods import *
import matplotlib.pyplot as plt


# Created by: Jiaxin Zhang (As one of authors or contributors of UQpy?)
# Author(Contributor???): Jiaxin Zhang
# Last modified by: 1/15/2018

# This is a new class - inference

# TODO: 16/56 todo list!
# 1. Information theoretic model selection √
# 2. Information theoretic multimodel selection √
# 3. Bayesian parameter estimation - Multimodel
# 4. Bayesian model selection
# 5. Bayesian parameter estimation - Conventional MLE
# 6. Optimal sampling density
# 7. Copula model selection
# 8. Copula multimodel selection
# 9. Copula parameter selection
# 10. Multimodel kriging??

# 11. Global Sensitivity Analysis (sampling class)
# 12. Importance sampling (sampling class)
# 13. Partially Stratified Sampling (sampling class)
# 14. Latinized Stratifed Sampling (sampling class)
# 15. Latinized Partially Stratified Sampling (sampling class)
# 16. Optimal sampling density (sampling or inference class)

# TODO：01-15-2018
# 1. using sampleMethods MCMC class or Ensemble MCMC
# 3. Bayesian prior information input
# 4. Bayesian model selection debug
# 5. Multimodel information theoretic selection
# 6. Bayesian parameter estimation - Conventional MLE


class Inference:

    ########################################################################################################################
    #                                        Information theoretic model selection - AIC, BIC
    ########################################################################################################################
    class InforMS:

        def __init__(self, data=None, model=None):
            """
            Created by: Jiaxin Zhang
            Last modified by: 1/03/2017
            Last modified by: 1/15/2017
            Last modified by: 3/29/2017

            """
            # TODO: more candidate probability models:
            # normal, lognormal, gamma, inverse gaussian, logistic, cauchy, exponential
            # weibull, loglogistic ??

            n = len(data)
            if model == 'normal':
                fitted_params_norm = stats.norm.fit(data)
                loglike = np.sum(stats.norm.logpdf(data, loc=fitted_params_norm[0], scale=fitted_params_norm[1]))
                k = 2

            elif model == 'cauchy':
                fitted_params_cauchy = stats.cauchy.fit(data)
                loglike = np.sum(stats.cauchy.logpdf(data, loc=fitted_params_cauchy[0], scale=fitted_params_cauchy[1]))
                k = 2

            elif model == 'exponential':
                fitted_params_expon = stats.expon.fit(data)
                loglike = np.sum(stats.expon.logpdf(data, loc=fitted_params_expon[0], scale=fitted_params_expon[1]))
                k = 2

            elif model == 'lognormal':
                fitted_params_logn = stats.lognorm.fit(data)
                loglike = np.sum(stats.lognorm.logpdf(data, s=fitted_params_logn[0], loc=fitted_params_logn[1], scale=fitted_params_logn[2]))
                k = 3

            elif model == 'gamma':
                fitted_params_gamma = stats.gamma.fit(data)
                loglike = np.sum(stats.gamma.logpdf(data, a=fitted_params_gamma[0], loc=fitted_params_gamma[1], scale=fitted_params_gamma[2]))
                k = 3

            elif model == 'invgauss':
                fitted_params_invgauss = stats.invgauss.fit(data)
                loglike = np.sum(stats.invgauss.logpdf(data, mu=fitted_params_invgauss[0], loc=fitted_params_invgauss[1], scale=fitted_params_invgauss[2]))
                k = 3

            elif model == 'logistic':
                fitted_params_logistic = stats.logistic.fit(data)
                loglike = np.sum(stats.logistic.logpdf(data, loc=fitted_params_logistic[0], scale=fitted_params_logistic[1]))
                k = 2

            aic_value = 2 * n - 2 * (loglike)
            aicc_value = 2 * n - 2 * (loglike) + (2*k**2 + 2*k)/(n-k-1)
            bic_value = np.log(n)*k - 2 * (loglike)

            self.AIC = aic_value
            self.AICC = aicc_value
            self.BIC = bic_value

    ########################################################################################################################
    #                                        Multimodel Information-theoretic selection
    ########################################################################################################################
    class MultiMI:
        def __init__(self, data=None, model=None):

            value = np.zeros((len(model)))
            model_sort = ["" for x in range(len(model))]

            for i in range(len(model)):
                model0 = model[i]
                value[i] = Inference.InforMS(data=data, model=model0).AICC

            v_sort = np.sort(value)
            # print(v_sort)
            sort_index = sorted(range(len(value)), key=value.__getitem__)
            for i in range((len(value))):
                s = sort_index[i]
                model_sort[i] = model[s]

            v_delta = v_sort - np.min(v_sort)

            s_AIC = 1.0
            w_AIC = np.empty([len(model), 1], dtype=np.float16)
            w_AIC[0] = 1.0

            delta = np.empty([len(model), 1], dtype=np.float16)
            delta[0] = 0.0

            for i in range(1, len(model)):
                delta[i] = v_sort[i] - v_sort[0]
                w_AIC[i] = np.exp(-delta[i] / 2)
                s_AIC = s_AIC + w_AIC[i]

            weights = w_AIC / s_AIC

            self.AICC = v_sort
            self.weights = weights
            self.delta = v_delta
            self.model_sort = model_sort

    ########################################################################################################################
    #                                         Bayesian inference - parameter estimation
    ########################################################################################################################
    class Bayes_Inference:
        """
        Created by: Jiaxin Zhang
        Last modified by: 3/29/2017

        """
        # TODO: prior type - noninformative, informative
        # TODO: prior distribution type - uniform, beta, normal etc.
        # TODO: MCMC algorithms - emcee

        def __init__(self, data=None, model=None):

            def lnlike(theta, data=data):
                if model == 'normal':
                    loglike = np.sum(stats.norm.logpdf(data, loc=theta[0], scale=theta[1]))
                    k = 2
                elif model == 'cauchy':
                    loglike = np.sum(stats.cauchy.logpdf(data, loc=theta[0], scale=theta[1]))
                    k = 2
                elif model == 'exponential':
                    loglike = np.sum(stats.expon.logpdf(data, loc=theta[0], scale=theta[1]))
                    k = 2
                elif model == 'lognormal':
                    loglike = np.sum(stats.lognorm.logpdf(data, s=theta[0], loc=theta[1], scale=theta[2]))
                    k = 3
                elif model == 'gamma':
                    loglike = np.sum(stats.gamma.logpdf(data, a=theta[0], loc=theta[1], scale=theta[2]))
                    k = 3
                elif model == 'invgauss':
                    loglike = np.sum(stats.invgauss.logpdf(data, mu=theta[0], loc=theta[1], scale=theta[2]))
                    k = 3
                elif model == 'logistic':
                    loglike = np.sum(stats.logistic.logpdf(data, loc=theta[0], scale=theta[1]))
                    k = 2

                return loglike

            def lnprior(theta):
                m, s = theta
                if -10000.0 < m < 10000.0 and 0.0 < s < 10000.0:
                    return 0.0
                return -np.inf
                #return 0.0

            def lnprob(theta, data=data):
                lp = lnprior(theta)
                if not np.isfinite(lp):
                    return -np.inf

                #return np.exp(lp + lnlike(theta, data))
                return (lp + lnlike(theta, data))

            # TODO: computing the evidence using numerical integration; monte carlo; RJMCMC?

            def integrate_posterior_2D(lnprob, xlim, ylim, data=data):
                func = lambda theta1, theta0: np.exp(lnprob([theta0, theta1], data))
                return integrate.dblquad(func, xlim[0], xlim[1], lambda x: ylim[0], lambda x: ylim[1])

            nwalkers = 50
            ndim = 2
            p0 = [np.random.rand(ndim) for i in range(nwalkers)]

            #target_like = np.exp(lnprob)

            # # using MCMC from samplemethods
            # MCMC = SampleMethods.MCMC(nsamples = 5000, dim = 2, x0 = [2,1], MCMC_algorithm = 'MH', proposal = 'Normal',
            #  params = np.ones(2), target = lnprob, njump = 1, marginal_parameters = [[0, 1], [0, 1]])
            # print(MCMC.samples)
            # plt.plot(MCMC.samples[:,0], MCMC.samples[:,1], 'ro')
            # plt.show()

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data])
            sampler.run_mcmc(p0, 500)
            trace = sampler.chain[:, 50:, :].reshape((-1, ndim))

            #fig = corner.corner(trace, labels=["$m$", "$s$"], truths=[0, 1, np.log(1)])
            #fig.show()
            #fig.savefig("pos_samples.png")
            # plt.plot(trace[:, 0], trace[:, 1], 'ko')
            # plt.show()

            #Bayesian parameter estimation - Conventional MLE
            # print(trace)
            # print(trace[0])
            loglike = np.zeros((len(trace)))
            for j in range(len(trace)):
                loglike[j] = lnlike(theta=trace[j], data=data)

            index = np.argmax(loglike)
            print(model)
            mle_Bayes = trace[index]
            print('MLE_Bayes:', mle_Bayes)

            # Bayes factor
            xlim, ylim = zip(trace.min(0), trace.max(0))

            Z1, err_Z1 = integrate_posterior_2D(lnprob, xlim, ylim)
            #print("Z1 =", Z1, "+/-", err_Z1)

            self.samples = trace
            self.Bayes_factor = Z1
            self.Bayes_mle = mle_Bayes

    ########################################################################################################################
    #                                         Multimodel Bayesian inference
    ########################################################################################################################
    class MultiMBayesI:
        def __init__(self, data=None, model=None):
            evi_value = np.zeros(len(model))
            mle_value = np.zeros(len(model))
            model_sort = ["" for x in range(len(model))]
            for i in range(len(model)):
                model0 = model[i]
                evi_value[i] = Inference.Bayes_Inference(data=data, model=model0).Bayes_factor
                # mle_value[i, :] = Inference.Bayes_Inference(data=data, model=model0).Bayes_mle

            sum_evi_value = np.sum(evi_value)
            nevi_value = -evi_value

            sort_index = sorted(range(len(nevi_value)), key=nevi_value.__getitem__)
            for i in range((len(nevi_value))):
                s = sort_index[i]
                model_sort[i] = model[s]

            bms = -np.sort(nevi_value)/sum_evi_value

            self.model_sort = model_sort
            self.weights = bms
            # self.mle_Bayes = mle_value