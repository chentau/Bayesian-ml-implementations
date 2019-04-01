import torch
import torch.distributions


def generate_data(n, k):
    datapoints = []
    mu = torch.distributions.MultivariateNormal(torch.zeros(k), 10*torch.eye(k)).sample()
    for _ in range(0, n):
        c = torch.distributions.Multinomial(torch.rep(1/k, k)).sample()
        datapoints.append(torch.distributions.Normal((mu*c).sum(), 1).sample())
        
