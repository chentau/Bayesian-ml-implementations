import numpy as np

from numpy import log
from scipy.special import gammaln, multigammaln
from scipy.special import logsumexp
from numpy.linalg import slogdet, inv
from numpy import log
from math import pi

LOGPI = log(pi)
np.seterr(all = "raise")

class CRP:
    def __init__(self, alpha, m, nu, kappa, Lambda):
        self.alpha = alpha
        self.m = m
        self.nu = nu
        self.kappa = kappa
        self.Lambda = Lambda

    def posterior_hyperparameters(self, data):
        """
        Calculates the posterior for the parameters of a Gaussian Inverse Wishart
        Prior, given [data]
        """
        n = data.shape[0]
        kappa_post = self.kappa + n
        nu_post = self.nu + n
        m_post = (self.kappa * self.m + n * data.mean(axis=0)) / \
                kappa_post
        Lambda_post = self.Lambda + np.dot(data.T, data) + \
                self.kappa * np.outer(self.m, self.m) - \
                kappa_post * np.outer(m_post, m_post)

        return nu_post, kappa_post, m_post, Lambda_post

    def marginal_lp(self, x):
        """
        returns the marginal likelihood of a single datapoint
        """
        nu_post, kappa_post, m_post,\
                Lambda_post = self.posterior_hyperparameters(x[None, :])

        lp = -.5 * self.dims * LOGPI
        lp += (.5 * self.dims) * log(self.kappa) - (.5 * self.dims) * log(kappa_post)
        lp += (.5 * self.nu) * slogdet(self.Lambda)[1] -\
                (.5 * nu_post) * slogdet(Lambda_post)[1]
        lp += multigammaln(.5 * nu_post, self.dims) -\
                multigammaln(.5 * self.nu, self.dims)
        return lp

    # def marginal_lp(self, x):
    #     nu_t = self.nu - self.dims + 1
    #     Lambda_t = (self.kappa + 1) / (self.kappa * nu_t) * self.Lambda
    #     Lambda_t_inv = inv(Lambda_t)

    #     lp = gammaln((nu_t + self.dims) / 2) - gammaln(nu_t / 2)
    #     lp -= .5 * self.dims * (log(nu_t) + LOGPI)
    #     lp -= .5 * slogdet(Lambda_t)[1]
    #     lp -= .5 * (nu_t + self.dims) * np.log(1 + (1 / nu_t) * 
    #     np.dot((x - self.m), Lambda_t_inv).dot((x - self.m)))
    #     return lp

    def lp_z_given_Z(self, k):
        n_k = self.clusters[k].n_points
        return log(n_k) - log(self.n + self.alpha - 1)

    def lp_z_new(self):
        return log(self.alpha) - log(self.n + self.alpha - 1)

    def log_likelihood(self):
        """
        The EPPF of the CRP
        """
        lp = self.k * log(self.alpha)
        for k in range(self.k):
            lp += log(np.math.factorial((self.clusters == k).sum()))
            if (self.clusters == k).sum() != 0:
                lp += self.marginal_likelihood(self.clusters[self.clusters == k])
        for i in range(1, self.n + 1):
            lp -= log(i - 1 + self.alpha)
        return lp

    def renormalize_cluster_assignments(self):
        """
        given an array of integer cluster assignments,
        re-normalize the assignments such that the assignments span from 1...k
        """
        sorted_assignments = np.unique(np.sort(self.cluster_assignments))
        normalized_assignments = np.arange(sorted_assignments.shape[0])
        normalized_dict = dict(zip(sorted_assignments, normalized_assignments))

        self.cluster_assignments = np.array([normalized_dict[o] for o in 
            self.cluster_assignments])

    def sample(self, data, n_iter=1000):
        self.data = data
        self.n = data.shape[0]
        
        try:
            self.dims = data[0].shape[0]
        except:
            self.dims = 1

        self.cluster_assignments = np.random.randint(np.ceil(self.alpha * log(self.n)),
                size=self.n)
        self.renormalize_cluster_assignments()
        self.k = np.max(self.cluster_assignments) + 1

        self.clusters = [Node(self.m, self.nu, self.kappa, self.Lambda,
            self.dims, data[self.cluster_assignments == k]) for k in range(self.k)]

        self.lp = np.zeros(n_iter)

        for t in range(n_iter):
            if t % 200 == 0:
                print("Iteration ", t)
                print("number of clusters: ", self.k)

            for i in range(self.n):
                # self.renormalize_cluster_assignments()

                # for k in range(self.k):
                #     assert self.clusters[k].n_points == (self.cluster_assignments
                #             == k).sum()

                ix = self.cluster_assignments[i]
                self.clusters[ix].remove_datapoint(data[i])

                # remove all empty clusters
                k_tmp = 0
                while (k_tmp < self.k):
                    if self.clusters[k_tmp].n_points == 0:
                        self.clusters.pop(k_tmp)
                        self.k -= 1
                        self.cluster_assignments[self.cluster_assignments > k_tmp] -= 1
                    else:
                        k_tmp += 1

                temp = np.zeros(self.k + 1)
                for k in range(self.k):
                    temp[k] = self.lp_z_given_Z(k) +\
                            self.clusters[k].predictive_lp(data[i])

                temp[-1] = self.lp_z_new() + self.marginal_lp(data[i])
                temp_normalized = temp - logsumexp(temp)

                # Take all numbers less than a cutoff to be 0
                # and renormalize, to prevent underflow
                temp_normalized[temp_normalized < -100] = -100
                temp_normalized = temp_normalized - logsumexp(temp_normalized)

                k_new = np.random.choice(self.k + 1, p=np.exp(temp_normalized))
                # cumsum_temp = np.array([logsumexp(temp_normalized[:i])
                #     for i in range(1, self.k + 2)])
                # u = log(np.random.uniform())
                # k_new = np.max(np.where(u <= cumsum_temp))

                if k_new < self.k:
                    self.cluster_assignments[i] = k_new
                    self.clusters[k_new].add_datapoint(data[i])
                else:
                    self.k += 1
                    self.clusters.append(Node(self.m, self.nu, self.kappa, 
                        self.Lambda, self.dims, data=data[i][None, :]))
                    self.cluster_assignments[i] = k_new

            # self.lp[t] = self.log_likelihood()

class Node(CRP):
    def __init__(self, m, nu, kappa, Lambda, dims, data=[]):
        self.dims = dims

        self.m = m
        self.nu = nu
        self.kappa = kappa
        self.Lambda = Lambda

        if len(data) == 0:
            self.n_points = 0

            self.m_post = m
            self.nu_post = nu
            self.kappa_post = kappa
            self.Lambda_post = Lambda
        else:
            self.nu_post, self.kappa_post, self.m_post,\
                    self.Lambda_post = super().posterior_hyperparameters(data)
            self.n_points = data.shape[0]

    def add_datapoint(self, x):
        self.n_points += 1

        self.kappa_post += 1
        self.nu_post += 1

        temp_m = self.m_post
        self.m_post = ((self.kappa_post - 1) * self.m_post + x) / self.kappa_post
        self.Lambda_post += np.outer(x, x) + (self.kappa_post - 1) *\
                np.outer(temp_m, temp_m) -\
                self.kappa_post * np.outer(self.m_post, self.m_post)

    def remove_datapoint(self, x):
        self.n_points -= 1

        self.kappa_post -= 1
        self.nu_post -= 1
        temp_m = self.m_post
        self.m_post = ((self.kappa_post + 1) * self.m_post - x) / self.kappa_post
        self.Lambda_post += (-1) * np.outer(x, x) - self.kappa_post *\
                np.outer(self.m_post, self.m_post) +\
                (self.kappa_post + 1) * np.outer(temp_m, temp_m)

    def predictive_lp(self, x):
        nu_t = self.nu_post - self.dims + 1
        Lambda_t = ((self.kappa_post + 1) / (self.kappa_post * nu_t)) *\
                self.Lambda_post
        Lambda_t_inv = inv(Lambda_t)

        lp = gammaln((nu_t + self.dims) / 2) - gammaln(nu_t / 2)
        lp -= .5 * self.dims * (log(nu_t) + LOGPI)
        lp -= .5 * slogdet(Lambda_t)[1]
        lp -= .5 * (nu_t + self.dims) * np.log(1 + (1 / nu_t) * 
        np.dot((x - self.m_post), Lambda_t_inv).dot((x - self.m_post)))
        return lp

