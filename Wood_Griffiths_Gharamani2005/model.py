import numpy as np
import scipy.stats as ss
import scipy.special

def P_z_given_Z_other(Z_col, row):
    """
    P(z_ij = 1 | Z_-ij)
    """
    m = Z_col[0:row].sum() + Z_col[row+1:].sum()
    theta = m/Z_col.size
    return theta

def P_X_given_others_Z(X, Y, Z, epsilon, lambd, row):
    t1 = np.matmul(Z[row, :], Y)
    t2 = 1 - (1-lambd)**(t1) * (1-epsilon)
    return (X[row, :] * np.log(t2) + (1 - X[row, :]) * np.log(1 - t2)).sum()

def P_z_given_others(X, Y, Z, epsilon, lambd, row, col):
    """
    P(z_ij = 1 | Z_-ij, Y, X)
    """
    t1 = np.log(P_z_given_Z_other(Z[:, col], row))
    z_val = Z[row, col]
    Z[row, col] = 1
    t2 = P_X_given_others_Z(X, Y, Z, epsilon, lambd, row)
    Z[row, col] = 0
    t3 = np.log(1 - P_z_given_Z_other(Z[:,col], row))
    t4 = P_X_given_others_Z(X, Y, Z, epsilon, lambd, row)
    Z[row, col] = z_val
    return np.exp(t1 + t2)/(np.exp(t1 + t2) + np.exp(t3 + t4))

def sample_z(X, Y, Z, epsilon, lambd, row, col):
    u = ss.uniform().rvs()
    if u < P_z_given_others(X, Y, Z, epsilon, lambd, row, col):
        return 1
    else:
        return 0
        
def P_X_given_others_Y(X, Y, Z, epsilon, lambd, col):
    """
    P(X_1:N,j = 1|Y, Z)
    """
    t1 = np.matmul(Z, Y[:,col])
    t2 = 1 - (1-lambd)**(t1) * (1-epsilon)
    return (X[:, col] * np.log(t2)).sum() + ((1 - X[:, col]) * np.log(1 - t2)).sum()

def P_y_given_others(X, Y, Z, epsilon, lambd, p, row, col):
    t1 = np.log(p)
    y_val = Y[row, col]
    Y[row, col] = 1
    t2 = P_X_given_others_Y(X, Y, Z, epsilon, lambd, col)
    t3 = np.log(1 - p)
    Y[row, col] = 0
    t4 = P_X_given_others_Y(X, Y, Z, epsilon, lambd, col)
    Y[row, col] = y_val
    return np.exp(t1 + t2)/(np.exp(t1 + t2) + np.exp(t3 + t4))

def sample_y(X, Y, Z, epsilon, lambd, p, row, col):
    u = ss.uniform().rvs()
    if u < P_y_given_others(X, Y, Z, epsilon, lambd, p, row, col):
        return 1
    else:
        return 0

def P_Knew(K, k_new, N, alpha):
    return ss.poisson(alpha/N).logpmf(K + k_new)

def P_X_given_others_K(X, Y, Z, k_new, epsilon, lambd, p, row):
    t1 = np.matmul(Z[row, :], Y)
    t2 = 1 - ((1-epsilon) * (1 - lambd)**t1 * (1-lambd*p)**k_new)
    return (X[row, :] * np.log(t2)).sum() + ((1 - X[row, :]) * np.log(1 - t2)).sum()

def P_Knew_given_others(X, Y, Z, K, k_new, epsilon, lambd, alpha, p, row):
    return P_X_given_others_K(X, Y, Z, k_new, epsilon, lambd, p, row) + \
            P_Knew(K, k_new, Z.shape[0], alpha)

def sample_K(X, Y, Z, K, epsilon, lambd, alpha, p, row):
    k_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    k_probs = np.array(list(map(lambda k:P_Knew_given_others(X, Y, Z, K, k, 
        epsilon, lambd, alpha, p, row), k_values)))
    k_probs = np.exp(k_probs)/(np.exp(k_probs).sum())
    k_cdf = np.array(list(map(lambda i: k_probs[0:i].sum(), range(1,12))))
    u = ss.uniform().rvs()
    K_new = min(np.where(u <= k_cdf, [k_values], [10])[0])
    return K_new

def log_joint(X, Y, Z, epsilon, lambd, alpha, p):
    t1 = np.matmul(Z, Y)
    t2 = 1 - (1 - lambd)**t1 * (1 - epsilon)
    lp_x = (X * np.log(t2) + (1 - X) * np.log(1 - t2)).sum()
    lp_y = (Y * np.log(p) + (1 - Y) * np.log(1 - p)).sum()
    
    N = Z.shape[0]
    mk = Z.sum(axis=0)
    k_plus = (mk > 0).sum()
    Hn = 1 / np.arange(1, N + 1).sum()
    lp_z = k_plus * np.log(alpha) - alpha*Hn + \
            (scipy.special.gammaln(N - mk + 1) + scipy.special.gammaln(mk) - scipy.special.gammaln(N+1)).sum()
    return lp_x + lp_y + lp_z

