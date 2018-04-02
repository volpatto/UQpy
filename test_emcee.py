import numpy as np
import emcee
from modelist import *
import corner

mu, sigma = 0, 1
data = np.random.normal(mu, sigma, 100)
print(data)


def lnlike(data, theta):
    loglike = np.sum(stats.norm.logpdf(data, loc=theta[0],scale=theta[1]))
    return loglike

def lnprior(theta):
    m,s = theta
    if -5.0 < m < 5.0 and 0.0 < s < 5.0:
        return 0.0
    return -np.inf

def lnprob(theta, data):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp+lnlike(data,theta)


nwalkers = 50
ndim = 2
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data])
sampler.run_mcmc(p0, 1000)
samples = sampler.chain[:, 10:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=["$m$", "$s$"],
                      truths=[0, 1, np.log(1)])
fig.savefig("triangle.png")



# def lnprob(x):
#     return -0.5 * np.sum(2 * x ** 2)
#
# ndim, nwalkers = 1, 10
# p0 = [np.random.rand(ndim) for i in range(nwalkers)]
#
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
# pos, prob, state = sampler.run_mcmc(p0, 20)
