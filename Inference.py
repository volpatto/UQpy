"""Design of Experiment methods. """
from library import *
import scipy.stats as stats
import numpy as np
from modelist import *
import os
import sys
import copy
from scipy.spatial.distance import pdist

# this is a new class including


class Inference:

    def __init__(self, data=None, model=None, method=None):
        self.method = method
        self.model = model
        self.data = data

    ########################################################################################################################
    #                                         Information Model Selection - AIC, BIC
    ########################################################################################################################
    class AIC:

        def __init__(self, data=None, model=None):
            """
            Created by: Jiaxin Zhang
            Last modified by: 12/03/2017
            """
            k = len(data)
            if model == 'normal':
                fitted_params_norm = stats.norm.fit(data)
                loglike = np.sum(stats.norm.logpdf(data, loc=fitted_params_norm[0],scale=fitted_params_norm[1]))

            elif model == 'cauchy':
                fitted_params_cauchy = stats.cauchy.fit(data)
                loglike = np.sum(stats.cauchy.logpdf(data, loc=fitted_params_cauchy[0],scale=fitted_params_cauchy[1]))

            elif model == 'exponential':
                fitted_params_expon = stats.expon.fit(data)
                loglike = np.sum(stats.expon.logpdf(data, loc=fitted_params_expon[0],scale=fitted_params_expon[1]))

            aic_value = 2 * k - 2 * (loglike)
            print(aic_value)
            self.aic = aic_value



