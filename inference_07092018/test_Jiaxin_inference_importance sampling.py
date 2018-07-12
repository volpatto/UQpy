import numpy as np
from Inference import ModelSelection
import matplotlib.pyplot as plt
import scipy.stats as stats


data = np.array([0.25023453,  2.3426804,   3.1530358,   1.74756396,  2.98132079,  2.51421884,
                 2.22117967,  0.92995667,  1.81050417,  2.25500144])

# plt.hist(data)
# plt.show()


x = ModelSelection(data=data, candidate_model=['Normal', 'Lognormal'])

print('model', x.model)
print('AICc_value', np.transpose(x.AICC))
print('AICc_delta', np.transpose(x.delta))
print('AICc_weights', np.transpose(x.weights))


y = ModelSelection(method='BayesMS', data=data.reshape(-1, 1), candidate_model=['Normal', 'Lognormal'],
                   pdf_proposal_type='Uniform', pdf_proposal_scale=[1], nsamples=10000, algorithm='Stretch', jump=100,
                   walkers=50)

print('model', y.model)
print('Bayesian_weights', np.transpose(y.weights))
print('samples', y.samples)

# # import corner
# import matplotlib.pyplot as plt
# #
# # fig = corner.corner(y.samples, labels=["$m$", "$s$"], truths=[0, 1, np.log(1)])
# # fig.show()
# # fig.savefig("pos_samples_UQpy.png")
# plt.plot(y.samples[:, 0], y.samples[:, 1], 'ko')

# Optimal sampling density, created by Jiaxin Zhang, 07/12/2018

p0 = y.samples[0] # posterior samples for Normal
p1 = y.samples[1] # posterior samples for Logrnomal

total_N_dis = 100 # total target densities
candidate_N_dis = (total_N_dis * x.weights)
dis1_num = int(candidate_N_dis[0])
dis2_num = int(candidate_N_dis[1])


import random

# a = len(p0)
# b = int(candidate_N_dis[0])
# print(a)
# print(b)
# print(random.sample(range(0,len(p0)), b))

para0_list = random.sample(range(0, len(p0)), dis1_num)
para1_list = random.sample(range(0, len(p1)), dis2_num)

para0_value = p0[para0_list]
para1_value = p1[para1_list]

N_rs = 1000 # random samples from the mixture distribution
mix_samples = np.zeros(N_rs)
for i in range(0, N_rs):
    rs_seed = random.sample(range(0, total_N_dis), 1)
    if rs_seed < candidate_N_dis[0]:
        mix_samples[i] = np.random.normal(para0_value[rs_seed, 0], para0_value[rs_seed, 1], 1)
    if rs_seed > candidate_N_dis[0]:
        rs_seed = int(rs_seed - candidate_N_dis[0])
        mix_samples[i] = np.random.lognormal(para1_value[rs_seed, 0], para1_value[rs_seed, 1], 1)

print(mix_samples)

plt.figure()
plt.hist(mix_samples, bins=50, density=True)
# plt.show()
plt.savefig("hist_samples.png")

num_mix_samples = len(mix_samples)
all_pdf = np.zeros((num_mix_samples, total_N_dis))
# optimal sampling density
k = -1
for i in range(len(x.weights)):
    for j in range(int(candidate_N_dis[i])):
        k = k + 1
        if i == 0:
            all_pdf[:, k] = stats.norm.pdf(mix_samples, loc=para0_value[j, 0], scale=para0_value[j, 1])
        if i == 1:
            all_pdf[:, k] = stats.lognorm.pdf(mix_samples, s=para0_value[j, 0], scale=para0_value[j, 1])


opt_pdf = np.sum(all_pdf, 1)/total_N_dis
print(opt_pdf)

plt.figure()
plt.plot(mix_samples, opt_pdf, '.')
# plt.show()
plt.savefig("optimal_sampling_density.png")

## uncertainty propagation with reweighting
def plate_buckling(x1, x2, x3, x4, x5, x6):
    f = (2.1/np.sqrt(x1**2 * x3/(x2**2*x4))-0.9/(np.sqrt(x1**2*x3/(x2**2*x4))**2))*(1-0.75*x5/np.sqrt(x1**2*x3/(x2**2*x4)))*(1-2*x6*x2/x1)
    return f


x1 = 36*0.992
x2 = 1.05*0.75
x3 = mix_samples + 34
x4 = 29000*0.9875
x5 = 1.0*0.35
x6 = 1.0*5.25
f = plate_buckling(x1, x2, x3, x4, x5, x6)

all_pdf = np.zeros((num_mix_samples, total_N_dis))
w = np.zeros((num_mix_samples, total_N_dis))
k = -1
for i in range(len(x.weights)):
    for j in range(int(candidate_N_dis[i])):
        k = k + 1
        if i == 0:
            all_pdf[:, k] = stats.norm.pdf(mix_samples, loc=para0_value[j, 0], scale=para0_value[j, 1])
            w[:, k] = all_pdf[:, k]/opt_pdf
        if i == 1:
            all_pdf[:, k] = stats.lognorm.pdf(mix_samples, s=para0_value[j, 0], scale=para0_value[j, 1])
            w[:, k] = all_pdf[:, k] / opt_pdf

index = np.argsort(f)
sort_f = np.sort(f)
plt.figure()
for i in range(total_N_dis):
    ss = sum(w[:, i])
    ww = w[:, i]/ss
    xw = ww[index]
    yw = np.cumsum(xw)
    plt.plot(sort_f, yw)

plt.show()
#plt.savefig("uncertainty_propagation.png")
