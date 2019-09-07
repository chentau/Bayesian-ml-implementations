import numpy as np
from scipy.special import gammaln, multigammaln
from numpy.linalg import slogdet, inv
from scipy.misc import logsumexp
from numpy import log
from math import pi

LOGPI = log(pi)
np.seterr(all = "raise")

class CollapsedGibbs():
    def __init__(self, k, m, kappa, nu, Lambda, alpha):
        """
        Collapsed Gibbs sampler for a gaussian mixture model, 
        with model parameters integrated out.

        Params
        ------
        m: mean hyperparameter for the Gaussian Inverse Wishart Prior
        kappa: degrees of freedom
        nu: degrees of freedom
        Lambda: prior precision matrix hyperparameter
        alpha: CRP concentration parameter
        """
        self.k = k
        self.m = m
        self.kappa = kappa
        self.nu = nu
        self.Lambda = Lambda
        self.alpha = alpha

    def lp_z_given_Z(self, ix, k):
        num_k = (self.clusters == k).sum()
        return log(num_k + self.alpha / self.k) \
            - log(self.n + self.alpha - 1)

    def lp_x_given_X(self, x, component):
        """
        The predictive distribution for x is a student's t distribution, with transformed hyperparameters
        
        """
        if component.n_k == 0:
            # If we don't have any other datapoints in the cluster, compute the marginal likelihood instead
            lp = component.marginal_likelihood(x)
        else:
            # Otherwise, calculate the predictive likelihood conditioned on all other datapoints in the same cluster
            lp = component.predictive_likelihood(x)
        return lp
    
    def lp_z_given_ZX(self, ix, k):
        return self.lp_z_given_Z(ix, k) + self.lp_x_given_X(ix, k)
    
    def posterior_hyperparameters(self, data):
        """
        Calculates the posterior for the parameters of a Gaussian Inverse Wishart
        Prior for our data
        """
        n = data.shape[0]
        kappa_posterior = self.kappa + n
        nu_posterior = self.nu + n
        m_posterior = (self.kappa * self.m + n * data.mean(axis=0)) / \
                kappa_posterior
        Lambda_posterior = self.Lambda + np.dot(data.T, data) + \
                self.kappa * np.outer(self.m, self.m) - \
                kappa_posterior * np.outer(m_posterior, m_posterior)
                
        return nu_posterior, kappa_posterior, m_posterior, Lambda_posterior
    
    def predictive_likelihood(self, data, x):
        """
        Calculates the hyperparameters for the predictive distribution
        P(x | X) ~ T(mu, Sigma, nu)
        """
        nu_posterior, kappa_posterior, m_posterior, Lambda_posterior = self.posterior_hyperparameters(data)
                
        nu_t = nu_posterior - self.dims + 1
        Lambda_t = ((kappa_posterior + 1) / (kappa_posterior * nu_t)) * Lambda_posterior
        Lambda_t_inv = inv(Lambda_t)
        
        lp = gammaln((nu_t + self.dims) / 2) - gammaln(nu_t / 2) \
             - (self.dims / 2) * log(nu_t) - (self.dims / 2) * LOGPI
        lp -= .5 * slogdet(Lambda_t)[1]
        lp -= ((nu_t + self.dims) / 2) * (1 + (1 / nu_t) * 
        np.dot((x - m_posterior), Lambda_t_inv).dot((x - m_posterior)))
        return lp
    
    def marginal_likelihood(self, data):
        nu_posterior, kappa_posterior, m_posterior, Lambda_posterior = self.posterior_hyperparameters(data)
        
        lp = -.5 * data.shape[0] * self.dims * LOGPI
        lp += (.5 * self.dims) * log(self.kappa) - (.5 * self.dims) * log(kappa_posterior)
        lp += (.5 * self.nu) * slogdet(self.Lambda)[1] - (.5 * nu_posterior) * slogdet(Lambda_posterior)[1]
        lp += multigammaln(.5 * nu_posterior, self.dims) - multigammaln(.5 * self.nu, self.dims)
        return lp
    
    def log_likelihood(self):
        """
        Joint likelihood of p(x, z)
        """
        lp = gammaln(self.alpha) - gammaln(self.n + self.alpha)
        for k in range(self.k):
            n_k = (self.clusters == k).sum()
            if n_k > 0:
                lp += gammaln(n_k + self.alpha / self.k) - gammaln(self.alpha / self.k)
                lp += self.marginal_likelihood(self.data[self.clusters == k, :])
        return lp

    def sample(self, data, burn=500, num_iter = 2000):
        n = len(data)
        self.data = data
        self.n = data.shape[0]
        self.dims = data.shape[1]
        self.clusters = np.zeros(n)
        self.samples = np.zeros((num_iter, n))
        self.lp = np.zeros(num_iter)
        self.components = []
        
        for t in range(num_iter):
            for i in range(self.n):
                # temporarily zero out the ith datapoint cluster
                x = self.data[i]
                components[cluster[i]].remove(x)
                temp = np.zeros(self.k)
                for k in range(self.k):
                    temp[k] = self.lp_z_given_ZX(x, self.components[k])
                
                temp_normalized = temp - logsumexp(temp)
                temp_cumulative = np.array([logsumexp(temp_normalized[0:i])
                    for i in range(1, temp_normalized.shape[0] + 1)])
                u = log(np.random.uniform())
                try:
                    k_new = np.argmax(temp_cumulative[temp_cumulative <= u]) + 1
                except:
                    k_new = 0
                self.clusters[i] = k_new
            self.samples[t, :] = self.clusters
            self.lp[t] = self.log_likelihood()


