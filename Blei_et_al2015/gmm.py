import torch
import torch.distributions

"""
phi: K x N matrix of cluster assignments
mu: K x 1 vector of cluster means
x: N x 1 vector of data points
"""

def update_phi(m, s2, x):
    """
    Variational update for the mixture assignments c_i
    """
    a = torch.ger(x, m)
    b = (s2+m**2)*.5
    return torch.transpose(torch.exp(a-b), 0, 1)/torch.exp(a-b).sum(dim = 1)

def update_m(x, phi, sigma_sq):
    """
    Variational update for the mean of the mixture mean
    distribution mu
    """
    num = torch.matmul(phi, x)
    denom = 1/sigma_sq + phi.sum(dim = 1)
    return num/denom

def update_s2(phi, sigma_sq):
    """
    Variational update for the variance of the mixture mean
    distribution mu
    """
    return (1/sigma_sq + phi.sum(dim = 1))**(-1)

def compute_elbo(phi, m, s2, x, sigma_sq):
    # The ELBO
    t1 = -(2*sigma_sq)**(-1)*(m**2 + s2).sum() + .5*torch.log(s2).sum()
    t2 = -.5 * torch.matmul(phi, x**2).sum() + (phi*torch.ger(m, x)).sum() \
            -.5*(torch.transpose(phi, 0, 1)*(m**2 + s2)).sum() 
            
    t3 = torch.log(phi)
    t3[t3 == float("-Inf")] = 0 # Prevent underflow
    t3 = - (phi*t3).sum()
    return t1 + t2 + t3

def generate_data(n, k, sigma_sq):
    datapoints = torch.zeros(n)
    mu = torch.distributions.MultivariateNormal(torch.zeros(k), sigma_sq*torch.eye(k)).sample()
    print(mu)
    for i in range(0, n):
        c = torch.distributions.Multinomial(1, torch.tensor(np.repeat(1/k, k))).sample().float()
        datapoints[i] = torch.distributions.Normal(loc=mu.dot(c), scale=1).sample()
    return datapoints

def fit(data, k, sigma_sq, num_iter = 2000):
    n = len(data)
    m = torch.distributions.MultivariateNormal(torch.zeros(k), torch.eye(k)).sample()
    s2 = torch.tensor([torch.distributions.Exponential(5).sample() for _ in range(0,k)])
    phi = torch.zeros((k,n), dtype=torch.float32)
    elbo = torch.zeros(num_iter)
    for i in range(0, n):
        phi[:,i] = torch.distributions.Dirichlet(torch.from_numpy(np.repeat(1.0,k))).sample().float()
    for j in range(0, num_iter):
        phi = update_phi(m, s2, data)
        m = update_m(data, phi, sigma_sq)
        s2 = update_s2(phi, sigma_sq)
        elbo[j] = compute_elbo(phi, m, s2, data, sigma_sq)
    return (phi, m, s2, elbo)

