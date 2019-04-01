import numpy as np
import scipy.stats as ss
import model
import matplotlib.pyplot as plt

# Constants for model fitting
K_init = 3
N = 50
T = 30
alpha = 3  # Concentration parameter of the IBP
epsilon = .01  # Baseline probability of exhibiting a symptom
p = .5  # Probability that a given cause is present on that trial
lambd = .9   # Relative effect of each hidden cause

def init_Z():
    """
    Initialize the hidden cause matrix Z
    """
    Z = np.zeros((N, K_init))
    return Z

def init_Y():
    """
    Initialize Y, where Y_kt details whether a given cause k is 
    present on trial t or not.
    """
    Y = np.empty((K_init, T))
    for row in range(K_init):
        for col in range(T):
            Y[row, col] = ss.bernoulli(P).rvs()
    return Y

def epsilon_step(X, Y, Z, epsilon, lambd, alpha, p):
    lp = model.log_joint(X, Y, Z, epsilon, lambd, alpha, p)
    epsilon_new = epsilon + ss.norm(loc=0, scale=.075).rvs()
    if (epsilon_new > 0 and epsilon_new < 1):
        lp_new = model.log_joint(X, Y, Z, epsilon_new, lambd, alpha, p)
        proposal_likelihood = lp_new - lp
        u = np.log(ss.uniform.rvs())
        if u < proposal_likelihood:
            epsilon = epsilon_new
    return epsilon

def lambda_step(X, Y, Z, epsilon, lambd, alpha, p):
    lp = model.log_joint(X, Y, Z, epsilon, lambd, alpha, p)
    lambd_new = lambd + ss.norm(loc=0, scale=.075).rvs()
    if (lambd_new > 0 and lambd_new < 1):
        lp_new = model.log_joint(X, Y, Z, epsilon, lambd_new, alpha, p)
        proposal_likelihood = lp_new - lp
        u = np.log(ss.uniform.rvs())
        if u < proposal_likelihood:
            lambd = lambd_new
    return lambd

def p_sweep(Y):
    num_ones = Y.sum()
    num_zeros = Y.size - num_ones
    return ss.beta(num_ones + 1, num_zeros + 1).rvs()

def chain(X):
    K = K_init  # Cut-off for number of hidden causes
    
    epsilon = ss.beta(1,1).rvs()
    lambd = ss.beta(1,1).rvs()
    p = ss.beta(1,1).rvs()
    alpha = 3

    Z = np.zeros((N, K))
    Y = np.zeros((K, T))
    while True:
        for row in range(N):
            for col in range(K):
                row_sum = Z[row, 0:col].sum() + Z[row, col+1:].sum()
                # Z[row, col] = model.sample_z(X, Y, Z, row, col)
                if row_sum > 0:
                    Z[row, col] = model.sample_z(X, Y, Z, epsilon, lambd, row, col)
                else:
                    Z[row, col] = 0
            K_new = model.sample_K(X, Y, Z, K, epsilon, lambd, alpha, p, row)
            Z_new = np.zeros((N, K_new))
            Z_new[row, :] = 1
            Z = np.concatenate((Z, Z_new), axis=1)
            
            Y_new = np.zeros((K_new, T))
            Y = np.concatenate((Y, Y_new), axis=0)
            for row_new in range(K+1, K+K_new):
                for col_y in range(T):
                    Y[row_new, col_y] = model.sample_y(X, Y, Z, epsilon, lambd, p, row_new, col_y)
            K += K_new

        for row in range(K):
            for col in range(T):
                Y[row, col] = model.sample_y(X, Y, Z, epsilon, lambd, p, row, col)
        
        zeros = np.where(Z.sum(axis=0) > 0)[0]
        Z = Z[:, zeros]
        Y = Y[zeros, :]
        K = Z.shape[1]
        epsilon = epsilon_step(X, Y, Z, epsilon, lambd, alpha, p)
        lambd = lambda_step(X, Y, Z, epsilon, lambd, alpha, p)
        p = p_sweep(Y)
        print(model.log_joint(X, Y, Z, epsilon, lambd, alpha, p))
        yield (Z, Y, epsilon, lambd, p)


def sample(burn, num_iter, X):
    chain1 = chain(X)
    posterior_Z = [np.zeros((N, K_init))]
    posterior_Y = [np.zeros((K_init, T))]
    posterior_epsilon = np.zeros(num_iter)
    posterior_lambd = np.zeros(num_iter)
    posterior_p = np.zeros(num_iter)
    for _ in range(burn):
        next(chain1)
    for i in range(num_iter):
        Z_samp, Y_samp, epsilon_samp, lambd_samp, p_samp = next(chain1)
        posterior_Z.append(Z_samp)
        posterior_Y.append(Y_samp)
        posterior_epsilon[i] = epsilon_samp
        posterior_lambd[i] = lambd_samp
        posterior_p[i] = p_samp
    return {"Z": posterior_Z, "Y": posterior_Y, "epsilon":posterior_epsilon, "lambda":posterior_lambd}

def main():
    Z = ss.bernoulli(.5).rvs(N*K_init).reshape((N, K_init))
    Y = ss.bernoulli(.5).rvs(K_init*T).reshape((K_init, T))
    print(Z)
    print(Y)
    t = np.matmul(Z, Y)
    X = ss.bernoulli(p=(1 - (1-lambd)**t * (1-epsilon))).rvs()
    print(X)
    
    # Run inference
    num_iter = 500
    posterior = sample(100, num_iter, X)
    agreement = np.zeros(num_iter)
    x = np.linspace(0, 1, num_iter)
    for i in range(0, num_iter):
        t1 = np.matmul(posterior["Z"][i], posterior["Y"][i])
        X_pred = 1 - (1 - lambd)**(t1) * (1 - epsilon)
        X_pred[X_pred > .5] = 1
        X_pred[X_pred <= .5] = 0
        agreement[i] = np.mod(X_pred + X, 2).sum()/X.size
    print(agreement.mean())
    plt.plot(x, posterior["epsilon"])
    plt.show()
    plt.plot(x, posterior["lambda"])
    plt.show()
    plt.plot(x, agreement)
    plt.show()
