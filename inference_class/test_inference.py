#from library import *
import scipy.stats as stats
import numpy as np
#from modelist import *
import os
import sys
import copy
from scipy.spatial.distance import pdist
#from Inference import *
import random

random.seed(31415926)

# data = [11624, 9388, 9471, 8927, 10865, 7698, 11744, 9238, 10319, 9750, 11462, 7939]
mu, sigma = 2, 1
data = np.random.normal(mu, sigma, 100)
#print(len(data))

model = 'normal'
#
# ### Information model selection using AIC and BIC
# Information_MS = Inference.InforMS(data=data, model=model)
#
# AIC_value = Information_MS.AIC
# BIC_value = Information_MS.BIC
# print('AIC:', AIC_value)
# print('BIC:', BIC_value)

# ### Bayesian inference
# BI = Inference.Bayes_Inference(data=data, model=model)
# samples = BI.samples
# Bayesian_evi = BI.Bayes_factor
# print('Bayesian evidence:', Bayesian_evi)

## multimodel information inference
#normal, lognormal, gamma, inverse gaussian, logistic, cauchy, exponential
model = ['normal', 'cauchy', 'exponential']
# model = ['normal','cauchy', 'exponential', 'lognormal', 'invgauss', 'logistic', 'gamma']
MMI = Inference.MultiMI(data=data, model=model)
print('model', MMI.model_sort)
print('AICc_value', np.transpose(MMI.AICC))
print('AICc_delta', np.transpose(MMI.delta))
print('AICc_weights', np.transpose(MMI.weights))

### multimodel Bayesian model selection
model = ['normal', 'cauchy', 'exponential']
MBayesI = Inference.MultiMBayesI(data=data, model=model)
print('model', MBayesI.model_sort)
print('Multimodel_Bayesian_Inference_weights', MBayesI.weights)
# print('MLE_Bayes', MBayesI.mle_Bayes)
