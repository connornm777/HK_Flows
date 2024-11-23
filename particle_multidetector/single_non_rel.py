import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from multiprocessing import Pool
from tqdm import tqdm

# Constants and parameters
l = 500.0        # Given parameter
m = 1.0        # Mass
sigma = 1.0
Lambda = 1.0
p0 = np.array([0.0, 0.0, 0.0])  # Initial momentum
x0 = np.array([0.0, 0.0, 0.0])  # Initial position
T = 10.0       # Total time for integration

# Define A (constant)
A = sigma**2 + 1 / (2 * Lambda**2)

# Define function to compute alpha(t) for given distance d
def alpha_t(t, d):
    x = np.array([d, 0.0, 0.0])  # Assuming x_n - x_0 along x-axis
    x_diff = x0 - x
    C = 1 / Lambda**2 + (t**2) / (2 * A * m**2)
    D = (2 * p0) / Lambda**2 - (t / (A * m)) * x_diff
    D_squared = np.dot(D, D)
    E = np.dot(p0, p0) / Lambda**2 + np.dot(x_diff, x_diff) / (2 * A)
    exponent = (D_squared) / (4 * C) - E
    prefactor = l / (8 * np.pi**3 * m) * (1 / (A**(1.5) * C**(1.5)))
    alpha = prefactor * np.exp(exponent)
    return alpha

# Function to compute P(D)(T) for a given distance d
def compute_P_D(d):
    # Integrate alpha_t from t=0 to t=T
    alpha_integral, _ = quad(lambda t: alpha_t(t, d), 0, T, limit=100)
    # Compute P(D)(T)
    P_D = 1 - np.exp(-alpha_integral)
    return P_D

# Define a range of distances
d_values = np.linspace(0.0, 10.0, 100)  # From 0 to 10 units
P_D_values = []

# Use multiprocessing to speed up computations
if __name__ == '__main__':
    with Pool() as pool:
        results = list(tqdm(pool.imap(compute_P_D, d_values), total=len(d_values)))

    P_D_values = results

    # Plot P(D)(T) vs. starting distance
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, P_D_values, label='$P(D)(T)$')
    plt.title(f'Decay Probability at Time T={T}')
    plt.xlabel('Starting Distance |$x_n - x_0$|')
    plt.ylabel('Decay Probability $P(D)(T)$')
    plt.legend()
    plt.grid()
    plt.show()
