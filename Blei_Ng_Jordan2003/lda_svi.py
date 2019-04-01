import numpy as np
import scipy.stats as ss
from scipy.special import psi, gammaln

"""
phi: d x k x v
gamma: d x k
lambda: k x v
"""

def update_phi_d(gamma, expElogbeta):
    t1 = np.exp(psi(gamma) - psi(gamma.sum()))[:, np.newaxis]
    t2 = t1 * expElogbeta
    return t2/t2.sum(axis=0)

def update_gamma_d(phi_d, counts, alpha):
    return alpha + np.matmul(phi_d, counts)

def update_lambda_intermediate(phi_d, counts, D, eta):
    return eta + D * (counts * phi_d)

def step_lambda(lambda_prev, rho, lambda_hat):
    return (1 - rho) * lambda_prev + rho * lambda_hat

def step_rho(tau, kappa, t):
    return (t + tau)**(-kappa)

def main(doc_term, K, num_iter, eta, alpha, tau, kappa):
    D = doc_term.shape[0]
    V = doc_term.shape[1]
    gamma = np.repeat(1, D*K).reshape((D, K))
    lambd = ss.gamma(1, 1).rvs((K, V))
    phi = np.empty((D, K, V))
    rho = 1
    for t in range(num_iter):
        for d in range(D):
            counts = np.array(doc_term.iloc[d, :])
            gamma_d = gamma[d, :]
            phi_d = np.empty((K, V))
            
            expElogbeta = np.exp(psi(lambd) - psi(lambd.sum(axis=1))[:, np.newaxis])
            for i in range(5):
                phi_d = update_phi_d(gamma_d, expElogbeta)
                gamma_d = update_gamma_d(phi_d, counts, alpha)
            lambda_hat = update_lambda_intermediate(phi_d, counts, D, eta)
            gamma[d, :] = gamma_d
            phi[d, :, :] = phi_d
            lambd = step_lambda(lambd, rho, lambda_hat)
            rho = step_rho(tau, kappa, t)
    return (gamma, phi, lambd)

def log_likelihood(doc_term, gamma, phi, lambd):
    D, V = doc_term.shape
    z = np.empty((D, V))
    log_likelihood = 0
    lambd_normalized = lambd/(lambd.sum(axis=1)[:, np.newaxis])
    for d in range(D):
        phi_d = phi[d, :, :]
        n = doc_term.iloc[d, :].sum()
        z = np.argmax(phi_d, axis=0) # MAP topic assignments
        probs = lambd_normalized[z, np.arange(V)] # Extract lambda_{z_{dn}, v}
        log_likelihood += ss.bernoulli(n, probs).logpmf(np.array(doc_term.iloc[d, :]))
    return log_likelihood

def elbo(counts, gamma_d, phi_d, lambd, alpha, eta):
    K = phi_d.shape[0]
    V = phi_d.shape[1]
    t1 = psi(gamma_d + 1e-100) - psi(gamma_d.sum())
    t2 =  psi(lambd) - psi(lambd.sum(axis=1))[:, np.newaxis]
    
    t3 = (alpha - 1) * t1.sum() + K * gammaln(alpha) - gammaln(alpha*K)
    t4 = (eta - 1) * t2.sum() + K * V * gammaln(eta) - K * gammaln(eta*V)
    
    t5 = (np.matmul(phi_d, counts) * t1).sum()
    t6 = (counts * phi_d * t2).sum()
    
    t6 = ((gamma_d - 1) * t1).sum() + gammaln(gamma_d + 1e-100).sum() - gammaln(gamma_d.sum())
    t7 = (((lambd - 1) * t2).sum(axis=1) + gammaln(lambd).sum(axis=1) \
            - gammaln(lambd.sum(axis=1))).sum()
    t8 = (counts * phi_d * np.log(phi_d + 1e-100)).sum()
    print(t1, t2, t3,t4,t5,t6,t7,t8)
    return t3 + t4 + t5 + t6 + t7 + t8
    
    

