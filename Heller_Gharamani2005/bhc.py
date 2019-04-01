import numpy as np
import itertools as it

from scipy.special import multigammaln, gamma
from numpy.linalg import det
from math import pi


class BHC:
    
    def __init__(self, data):
        self.nodes = []
        n = len(data)
        for i, datapoint in enumerate(data):
            temp_node = Node(np.array([datapoint]), i)
            temp_node.loglikelihood = temp_node.logml
            self.nodes.append(temp_node)
    
    def _merge(self, node1, node2):
        node_merge = Node(np.vstack((node1.data, node2.data)), 
                label = node1.label, children=[node1, node2])
        node_merge.loglikelihood = np.logaddexp(np.log(node_merge.prior) + node_merge.logml,
                np.log(1 - node_merge.prior) + node1.loglikelihood + node2.loglikelihood)
        node_merge.rk = np.exp(np.log(node_merge.prior) + node_merge.logml - node_merge.loglikelihood)
        assert(node_merge.rk <= 1)
        return node_merge
    
    def fit(self):
        while len(self.nodes) > 1:
            max_rk = 0
            for i, node1 in enumerate(self.nodes):
                for node2 in self.nodes[i+1:]:
                    temp_merge = self._merge(node1, node2)
                    
                    if temp_merge.rk > max_rk:
                        max_rk = temp_merge.rk
                        max_merge = temp_merge
                        # max_merge.rk = temp_merge.rk
                        max_node2 = node2
                        max_index = i

            self.nodes[max_index] = max_merge
            self.nodes.remove(max_node2)
 
class Node:
    
    def __init__(self, data, label, children = None, alpha=1, kappa=1, df=5):
        
        try:
            self.d = len(data[0])
        except TypeError:
            self.d = 1
        self.n = len(data)
        
        # Hyperparameters
        self.df = df
        self.kappa = kappa
        
        # xx^T, proportional to the covariance matrix
        if self.n == 1:
            self.scatter = np.zeros(self.d)
        else:
            self.scatter = np.cov(data.T) * (self.n - 1)
        
        self.data = data
        self.children = children
        self.label = label
        
        # P(data)
        self.logml = self._calculate_logml()
        
        # P(data | tree).  When there is only one datapoint, this is equal to self.logml
        self.loglikelihood = self.logml
        
        # Probability that the data in this node was generated from one cluster
        # Defaults to 1 for leafs
        self.rk = 1
        
        if not children:
            self.dk = alpha
            self.prior = 1
        else:
            self.dk = alpha*gamma(self.n) + self.children[0].dk * self.children[1].dk
            self.prior = alpha * gamma(self.n)/self.dk
        assert(self.prior <= 1)
        
    def _calculate_logml(self):
        """
        Marginal likelihood for continuous data, with a 
        Normal Inverse Wishart Prior - NIW(0, kappa, df, I)
        For derivation, see [Conjugate Bayesian Analysis of the Gaussian Distribution] - Murphy
        """
        score = -(self.n * self.d/2) * np.log(pi)
        score += multigammaln((self.df + self.n)/2, self.d) - multigammaln(self.df/2, self.d)
        posterior_cov = np.eye(self.n) + self.scatter + (self.kappa * self.n)/(self.kappa + self.n)*np.outer(self.data.mean(), self.data.mean())
        score -=  (self.df + self.n)/2 * np.log(abs(det(posterior_cov)))
        score += self.d/2 * (np.log(self.kappa) - np.log(self.kappa + self.n))
        if score > 0:
            print(self.data)
            print(score)
            raise RuntimeError("Log Marginal Likelihood should be negative!")
        return score
