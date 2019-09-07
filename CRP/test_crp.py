import pdb
import numpy as np
import scipy.stats as ss

from crp import CRP

t1 = ss.multivariate_normal([10,10], np.eye(2)).rvs(300)
# t2 = ss.multivariate_normal([5,5], np.eye(2)).rvs(300)
t3 = ss.multivariate_normal([-5,-5], np.eye(2)).rvs(300)

# data = np.concatenate([t1, t2, t3])
data = np.concatenate([t1, t3])


test = CRP(alpha=1, m=np.zeros(2), nu = 5, kappa = 5, Lambda = 50 * np.eye(2))

# pdb.set_trace()
test.sample(data, n_iter=400)
