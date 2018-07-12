from UQpy.Inference import ModelSelection
import numpy as np


data = np.array([0.25023453,  2.3426804,   3.1530358,   1.74756396,  2.98132079,  2.51421884,
                 2.22117967,  0.92995667,  1.81050417,  2.25500144])


x = ModelSelection(data=data, candidate_model=['rayleigh', 'exponential', 'Gamma',
                                               'logistic', 'pareto', 'inv_gauss'])


print('model', x.model)
print('AICc_value', np.transpose(x.AICC))
print('AICc_delta', np.transpose(x.delta))
print('AICc_weights', np.transpose(x.weights))


y = ModelSelection(method='BayesMS', data=data.reshape(-1, 1), candidate_model=['rayleigh'],
                   pdf_proposal_type='Uniform', pdf_proposal_scale=[1], nsamples=150000, algorithm='Stretch', jump=1000,
                   walkers=50)

print('model', y.model)
print('Bayesian_weights', np.transpose(y.weights))


print()
'''
import corner
import matplotlib.pyplot as plt

fig = corner.corner(y.samples, labels=["$m$", "$s$"], truths=[0, 1, np.log(1)])
fig.show()
fig.savefig("pos_samples_UQpy.png")
plt.plot(y.samples[:, 0], y.samples[:, 1], 'ko')
'''