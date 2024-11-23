# Import necessary libraries
import numpy as np
import cupy as cp  # CuPy for GPU computations
from cupyx.scipy.special import i0  # Modified Bessel function of the first kind, order zero
import matplotlib.pyplot as plt

# Define constants and parameters
# Physical constants
g = 1.0  # Coupling constant

# Lambda parameters (positive real constants)
Lambda1 = 0.1
Lambda2 = 0.1
Lambda3 = 0.1

# Momentum vectors (can be varied)
p01 = cp.array([1.0, 0.0, 0.0])
p02 = cp.array([-1.0, 0.0, 0.0])
p03 = cp.array([0.0, 0.0, 0.0])

# Position vectors (can be varied)
x01 = cp.array([0.0, 0.0, 0.0])
x02 = cp.array([0.0, 0.0, 0.0])
x03 = cp.array([0.0, 0.0, 0.0])

# Upper limit for numerical integration
L = 10.0

# Number of points for integration grid
num_points = 500  # Adjust for desired accuracy

# Parameters to vary and hold constant
vary_parameters = {
    'm': np.linspace(0.1, 5.0, 100),
    # Add other parameters here if needed
}

constant_parameters = {
    'g': g,
    'Lambda1': Lambda1,
    'Lambda2': Lambda2,
    'Lambda3': Lambda3,
    'p01': p01,
    'p02': p02,
    'p03': p03,
    'x01': x01,
    'x02': x02,
    'x03': x03,
    # Add other parameters here if needed
}

# Choose which parameter to vary
vary_param = 'm'
param_values = vary_parameters[vary_param]

# Function to compute |A|^2 for given parameters
def compute_A_squared(m, g, Lambda1, Lambda2, Lambda3, p01, p02, p03, x01, x02, x03):
    # Update coefficients
    a_p = 1 / Lambda1 + 1 / Lambda3
    a_k = 1 / Lambda2 + 1 / Lambda3
    b = 1 / Lambda3

    # Matrix M and its determinant
    detM = a_p * a_k - b**2

    # Inverse of M
    M_inv = (1 / detM) * cp.array([[a_k, -b],
                                    [-b, a_p]])

    # D_p and D_k as complex vectors
    D_p = (2 / Lambda1) * p01 + 1j * x01
    D_k = (2 / Lambda2) * p02 + 1j * x02

    # Constants from the exponent (C0)
    C0 = - (1 / Lambda1) * cp.dot(p01, p01) - (1 / Lambda2) * cp.dot(p02, p02)

    # Compute c^dagger M_inv c
    c_p_conj = cp.conj(D_p)
    c_k_conj = cp.conj(D_k)

    c_Minv_c = (
        M_inv[0, 0] * cp.dot(c_p_conj, D_p) +
        M_inv[0, 1] * cp.dot(c_p_conj, D_k) +
        M_inv[1, 0] * cp.dot(c_k_conj, D_p) +
        M_inv[1, 1] * cp.dot(c_k_conj, D_k)
    )

    exponential_term = cp.exp(0.25 * c_Minv_c + C0)

    # Compute the constant prefactor
    prefactor = (g / ((2 * np.pi)**(9/4) * (Lambda1 * Lambda2 * Lambda3)**(1.5)))**2
    prefactor /= detM

    # Create grid of p and k values
    p = cp.linspace(0, L, num_points)[1:]  # Exclude zero to avoid division by zero
    k = cp.linspace(0, L, num_points)[1:]

    # Create meshgrid for p and k
    P, K = cp.meshgrid(p, k, indexing='ij')

    # Compute cos(theta0)
    cos_theta0 = 1 - (m**2) / (2 * K * P)

    # Mask invalid values where cos_theta0 is outside [-1, 1]
    valid_mask = (cos_theta0 >= -1) & (cos_theta0 <= 1)

    # Initialize integrand array
    integrand = cp.zeros_like(P)

    # Compute sin(theta0)
    sin_theta0 = cp.sqrt(1 - cos_theta0**2)

    # Exponential term in the integrand
    expo = - (a_p * P**2 + a_k * K**2 + 2 * b * P * K) + (D_p[2] * P * cos_theta0).real + (D_k[2] * K).real

    # Compute the Bessel function argument
    D_p_perp = cp.sqrt(D_p[0].real**2 + D_p[1].real**2)
    Bessel_arg = P * sin_theta0 * D_p_perp

    # Compute the Bessel function
    I0_value = i0(Bessel_arg)

    # The prefactor
    prefactor_integrand = (K * P)**(1.5) / cp.sqrt(K + P)

    # Compute integrand where valid
    integrand[valid_mask] = prefactor_integrand[valid_mask] * cp.exp(expo[valid_mask]) * I0_value[valid_mask]

    # Numerically integrate over p and k using the trapezoidal rule
    integral_p = cp.trapz(integrand, k, axis=1)
    integral = cp.trapz(integral_p, p)

    # Compute |A|^2
    A_squared = (prefactor * exponential_term * integral)**2

    return A_squared.get().real  # Use .get() to bring result back to CPU

# List to store |A|^2 values for different parameter values
A_squared_values = []

# Loop over parameter values
for value in param_values:
    # Update the varying parameter
    if vary_param == 'm':
        m = value
        A_squared = compute_A_squared(m, **constant_parameters)
    else:
        # Handle other parameters if needed
        pass

    A_squared_values.append(A_squared)

    print(f"{vary_param} = {value:.2f}, |A|^2 = {A_squared:.6e}")

# Plotting |A|^2 as a function of the varying parameter
plt.figure(figsize=(8, 6))
plt.plot(param_values, A_squared_values, label=r'$|A|^2$')
plt.xlabel(f'{vary_param}')
plt.ylabel(r'$|A|^2$')
plt.title(rf'Plot of $|A|^2$ vs {vary_param}')
plt.legend()
plt.grid(True)
plt.show()
