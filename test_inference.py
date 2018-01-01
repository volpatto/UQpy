from library import *
import scipy.stats as stats
import numpy as np
from modelist import *
import os
import sys
import copy
from scipy.spatial.distance import pdist
from Inference import *

# data = handle['data']
# model = handle['model']
# method = handle['method']

data = [11624, 9388, 9471, 8927, 10865, 7698, 11744, 9238, 10319, 9750, 11462, 7939]
model = 'normal'

infe = Inference(data=data, model=model, method=None)

aic = infe.AIC(data=data, model=model)

print(aic)