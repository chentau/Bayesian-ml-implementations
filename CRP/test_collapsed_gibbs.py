import numpy as np
import scipy.stats as ss

import collapsed_gibbs

t1 = ss.multivariate_normal([0,0], [[1,0],[0,1]]).rvs(5)
t2 = ss.multivariate_normal([10,10], [[1,0],[0,1]]).rvs(5)
t3 = np.concatenate([t1, t2])
test = collapsed_gibbs.CollapsedGibbs(2, np.array([0,0]), 5, 5, np.eye(2), 1)

test.sample(t3)
