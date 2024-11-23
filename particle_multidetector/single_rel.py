import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
from multiprocessing import Pool
from tqdm import tqdm

# Constants and parameters
l = 1.0         # Coupling constant
m = 1.0         # Mass
sigma = 1.0
Lambda = 1.0
p0 = 1.0        # Initial momentum magnitude
x0 = 0.0        # Initial position
T = 10.0        # Total time for integration

# Define a range of starting distances
d_values = np.linspace(0.0, 10.0, 50)  # From 0 to 10 units

# Define function to compute S(p, k, d, t)
def S(p, k, d, t):
    E_p = np.sqrt(p**2 + m**2)
    E_k = np.sqrt(k**2 + m**2)
    Delta_x = x0 - d  # Assuming x_n is at position d along x-axis

    # Exponential terms
    term1 = -0.5 * sigma**2 * (p - k)**2
    term2 = -0.5 / Lambda**2 * ((p - p0)**2 + (k - p0)**2)
    term3 = -1j * (p - k) * Delta_x
    term4 = -1j * t * (E_p - E_k)
    S_total = term1 + term2 + term3 + term4
    return S_total

# Function to compute the integrand for alpha(t, d)
def integrand(p, k, d, t):
    E_p = np.sqrt(p**2 + m**2)
    E_k = np.sqrt(k**2 + m**2)
    S_total = S(p, k, d, t)
    integrand_value = (p**2 * k**2) / (E_p * E_k) * np.exp(S_total)
    return np.real(integrand_value)

# Function to compute alpha(t, d)
def alpha_t(t, d):
    # Integration limits
    p_min, p_max = -np.inf, np.inf
    k_min, k_max = -np.inf, np.inf

    # Perform double integration over p and k
    integral, error = dblquad(
        lambda p, k: integrand(p, k, d, t),
        k_min, k_max,  # Limits for k
        lambda k: p_min, lambda k: p_max  # Limits for p
    )
    alpha = l / (2 * np.pi)**6 * integral
    return alpha

# Function to compute P(D)(T) for a given distance d
def compute_P_D(d):
    # Integrate alpha_t from t=0 to t=T
    alpha_integral, _ = quad(lambda t: alpha_t(t, d), 0, T, limit=50)
    # Compute P(D)(T)
    P_D = 1 - np.exp(-alpha_integral)
    return P_D

# Compute P(D)(T) for all distances
if __name__ == '__main__':
    P_D_values = []

    with Pool() as pool:
        results = list(tqdm(pool.imap(compute_P_D, d_values), total=len(d_values)))

    P_D_values = results

    # Plot P(D)(T) vs. starting distance
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, P_D_values, label='$P(D)(T)$')
    plt.title(f'Decay Probability at Time T={T}')
    plt.xlabel('Starting Distance $d = |x_n - x_0|$')
    plt.ylabel('Decay Probability $P(D)(T)$')
    plt.legend()
    plt.grid()
    plt.show()
