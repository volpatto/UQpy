# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""This module contains functionality for all the Inference supported in UQpy."""


from SampleMethods import *
from scipy import integrate


########################################################################################################################
########################################################################################################################
#                            Information theoretic model selection - AIC, BIC
########################################################################################################################

class ModelSelection:

    def __init__(self,  data=None, candidate_model=None, method=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 algorithm=None, jump=None, nsamples=None, seed=None, nburn=None, parameters=None, walkers=None):

        self.data = data
        self.method = method
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.algorithm = algorithm
        self.jump = jump
        self.nsamples = nsamples
        self.nburn = nburn
        self.seed = seed
        self.parameters = parameters
        self.walkers = walkers
        self.candidate_model = candidate_model
        self.dist_models = list()
        for i in range(len(self.candidate_model)):
            self.dist_models.append(Distribution(self.candidate_model[i]))

        if self.method == 'InfoMS' or self.method is None:
            self.AICC, self.weights, self.delta, self.model, self.Parameters = self.multi_info_ms()
        elif self.method == 'BayesMS':
            self.model, self.weights, self.samples = self.multi_bayes_ms()

    def info_ms(self, model):

        n = self.data.shape[0]
        fit_i = model.fit
        params = list(fit_i(self.data))
        log_like, k = ln_like(model, params, self.data)

        aic_value = 2 * n - 2*log_like
        aicc_value = 2 * n - 2 * log_like + (2*k**2 + 2*k)/(n-k-1)
        bic_value = np.log(n)*k - 2 * log_like

        return aic_value, aicc_value, bic_value, list(params)

    def multi_info_ms(self):

            aicc = np.zeros((len(self.dist_models)))
            aic = np.zeros_like(aicc)
            bic = np.zeros_like(aicc)
            model_sort = ["" for x in range(len(self.dist_models))]
            params = [0]*len(self.dist_models)
            for i in range(len(self.dist_models)):
                print('Running Informative model selection...', self.dist_models[i].name)
                aic[i], aicc[i], bic[i], params[i] = self.info_ms(self.dist_models[i])

            v_sort = np.sort(aicc)
            params_sort = [0]*len(self.dist_models)
            sort_index = sorted(range(len(self.dist_models)), key=aicc.__getitem__)
            for i in range((len(self.dist_models))):
                s = sort_index[i]
                model_sort[i] = self.dist_models[s].name
                params_sort[i] = params[s]

            v_delta = v_sort - np.min(v_sort)

            s_aic = 1.0
            w_aic = np.empty([len(self.dist_models), 1], dtype=np.float16)
            w_aic[0] = 1.0

            delta = np.empty([len(self.dist_models), 1], dtype=np.float16)
            delta[0] = 0.0

            for i in range(1, len(self.dist_models)):
                delta[i] = v_sort[i] - v_sort[0]
                w_aic[i] = np.exp(-delta[i] / 2)
                s_aic = s_aic + w_aic[i]

            weights = w_aic / s_aic

            return v_sort, weights, v_delta, model_sort, params_sort

    def multi_bayes_ms(self):
        evi_value = np.zeros(len(self.dist_models))
        post_samples = list()
        for i in range(len(self.dist_models)):
            model0 = self.dist_models[i]
            fit_i = model0.fit
            dimension = len(list(fit_i(self.data)))

            if dimension > 3:
                raise RuntimeError('Only distributions with three-dimensional parameters can be solved.')

            self.seed = np.array([np.random.rand(dimension) for i in range(self.walkers)])

            x = BayesianInference(data=self.data, dimension=dimension, candidate_model=model0,
                                  nsamples=self.nsamples, parameters=[self.parameters],
                                  algorithm=self.algorithm, nburn=self.nburn,
                                  pdf_proposal_scale=self.pdf_proposal_scale,
                                  pdf_proposal_type=self.pdf_proposal_type,
                                  seed=self.seed)

            evi_value[i] = x.Bayes_factor
            post_samples.append(x.samples)

        sum_evi_value = np.sum(evi_value)
        nevi_value = -evi_value

        model_sort = ["" for r in range(len(self.dist_models))]
        sort_index = sorted(range(len(nevi_value)), key=nevi_value.__getitem__)
        for i in range((len(nevi_value))):
            s = sort_index[i]
            model_sort[i] = self.dist_models[s].name

        bms = -np.sort(nevi_value) / sum_evi_value

        return model_sort, bms, post_samples


########################################################################################################################
########################################################################################################################
#                                  Bayesian Inference
########################################################################################################################

class BayesianInference:

    def __init__(self, data=None, dimension=None, candidate_model=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 algorithm=None, jump=None, nsamples=None, seed=None, nburn=None, parameters=None):

        self.data = data
        self.parameters = parameters[0]
        self.dimension = dimension
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.algorithm = algorithm
        self.jump = jump
        self.nsamples = nsamples
        self.seed = seed
        self.nburn = nburn
        if isinstance(candidate_model, Distribution) is True:
            self.dist_model = candidate_model
        else:
            self.dist_model = Distribution(candidate_model[0])

        self.samples, self.Bayes_factor, self.Bayes_mle = self.bayes_inf()

    def bayes_inf(self):
        from SampleMethods import MCMC

        z = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                 pdf_proposal_scale=self.pdf_proposal_scale,
                 algorithm=self.algorithm, jump=self.jump, seed=self.seed, nburn=self.nburn, nsamples=self.nsamples,
                 pdf_target_type='joint_pdf', pdf_target=self.ln_prob, pdf_target_params=self.parameters)
        trace = z.samples

        log_like = np.zeros(trace.shape[0])
        for i in range(trace.shape[0]):
            log_like[i], k = ln_like(self.dist_model, trace[i], self.data)

        index = np.argmax(log_like)
        mle_bayes = trace[index]

        if self.dimension == 2:
            x_lim, y_lim = zip(trace.min(0), trace.max(0))
            z1, err_z1 = self.integrate_posterior_2d(x_lim, y_lim)
        elif self.dimension == 3:
            x_lim, y_lim, z_lim = zip(trace.min(0), trace.max(0))
            z1, err_z1 = self.integrate_posterior_3d(x_lim, y_lim, z_lim)

        return trace, z1, mle_bayes

    def integrate_posterior_2d(self,  x_lim, y_lim):
        fun_ = lambda theta1, theta0: np.exp(self.ln_prob([theta0, theta1], None))
        return integrate.dblquad(fun_, x_lim[0], x_lim[1], lambda x: y_lim[0], lambda x: y_lim[1])

    def integrate_posterior_3d(self, x_lim, y_lim, z_lim):
        fun_ = lambda theta2, theta1, theta0: np.exp(self.ln_prob([theta0, theta1, theta2], None))
        return integrate.tplquad(fun_, x_lim[0], x_lim[1], lambda x: y_lim[0], lambda x: y_lim[1],
                                 lambda x, y: z_lim[0], lambda x, y: z_lim[1])

    def ln_prob(self, theta, args):
        lp = ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        lnlik3, k = ln_like(self.dist_model, theta, self.data)
        return lp + lnlik3

########################################################################################################################
########################################################################################################################
#                                  Necessary functions
########################################################################################################################


def ln_like(model, theta, data):
    # One-dimensional case for the log-likelihood function
    ln_pdf = model.log_pdf
    k = len(theta)
    log_pdf = ln_pdf(data, theta)

    return np.sum(log_pdf), k


def ln_prior(params):
    #  Uniform Prior distribution
    # Log(1.0) = 0.0
    return 0.0


