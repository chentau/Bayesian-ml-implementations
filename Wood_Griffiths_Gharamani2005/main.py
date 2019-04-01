import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import sampler

# Constants for model fitting
K = 10  # Cut-off for number of hidden causes
T = 10  # Number of trials
N = 10  # Number of observations
alpha = 3  # Concentration parameter of the IBP
epsilon = .01  # Baseline probability of exhibiting a symptom
p = .1  # Probability that a given cause is present on that trial
lambd = .9   # Relative effect of each hidden cause

sampler.K = K
sampler.T = T
sampler.N = N
sampler.alpha = alpha
sampler.epsilon = epsilon
sampler.p = p
sampler.lambd = lambd


def generate_data():
    sticks = sampler.init_sticks()
    z = sampler.init_Z(sticks)
    y = sampler.ss.bernoulli(p).rvs(K*T).reshape((K, T))
    x = sampler.np.empty((N, T))
    for row in range(N):
        for col in range(T):
            x[row, col] = ss.bernoulli(p=(1 - (1-lambd)**(z[row, :] @ y[:, col])*(1-epsilon))).rvs()
    return (x, y, z)


def infer(data, burn, num_iter):
    posterior = sampler.sample(burn, num_iter, data)
    return posterior


def calculate_agreement(posterior, z):
    """
    Calculate |z' - z|/N*K, which is the absolute error between
    the inferred z' and the actual ground truth z.
    """
    num_iter = posterior["Z"].shape[2]
    size = posterior["Z"][:, :, 0].size
    agreement = np.zeros(num_iter)
    for i in range(num_iter):
        agreement[i] = np.abs(z - posterior["Z"][:, :, i]).sum()/size
    return agreement


def trace_plot_stick(posterior, index):
    stick = posterior["sticks"][index, :]
    plt.plot(np.linspace(0, 1, len(stick)), stick)
    plt.show()


def plot_agreement(agreement):
    plt.plot(np.linspace(0, 1, len(agreement)), agreement)
    plt.show()


if __name__ == "__main__":
    x, y, z = generate_data()
    posterior = infer(x, 0, 500)
    agreement = calculate_agreement(posterior, z)
    print(agreement.mean())
    print(agreement.var())
    #plot_agreement(agreement)
    # Plot the first 5 trace plots
    # for i in range(5):
    #     trace_plot_stick(posterior, i)
