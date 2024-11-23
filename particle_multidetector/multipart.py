import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Enable LaTeX rendering in Matplotlib
plt.rcParams['text.usetex'] = True


def compute_P_two_particle(
        p01, p02, x01, x02, sigma2=1.0, Lambda1=1.0, Lambda2=1.0, num_points=30, m=1.0, l=1.0
):
    """
    Compute P(D) for two particles with given parameters.

    Parameters:
    - p01: array-like, initial momentum of particle 1 (shape: (3,))
    - p02: array-like, initial momentum of particle 2 (shape: (3,))
    - x01: array-like, initial position of particle 1 (shape: (3,))
    - x02: array-like, initial position of particle 2 (shape: (3,))
    - sigma2: float, σ^2 parameter of the detector function
    - Lambda1: float, Λ parameter (width) of particle 1's wave packet
    - Lambda2: float, Λ parameter (width) of particle 2's wave packet
    - num_points: int, number of quadrature points for integration
    - m: float, mass of the particles
    - l: float, detector coupling strength

    Returns:
    - P: float, computed detection probability
    """
    # Convert inputs to CuPy arrays
    p01 = cp.asarray(p01)
    p02 = cp.asarray(p02)
    x01 = cp.asarray(x01)
    x02 = cp.asarray(x02)

    # Integration ranges for momenta
    p_min, p_max = -10.0, 10.0  # Adjust as needed

    # Generate quadrature points and weights
    p_points_np, p_weights_np = np.polynomial.legendre.leggauss(num_points)
    # Rescale points to the desired interval
    p_points_np = 0.5 * (p_points_np + 1) * (p_max - p_min) + p_min
    p_weights_np *= 0.5 * (p_max - p_min)
    # Convert to CuPy arrays
    p_points = cp.asarray(p_points_np)
    p_weights = cp.asarray(p_weights_np)

    # Generate grids for p1, p2, k1, k2
    P1, P2 = cp.meshgrid(p_points, p_points, indexing='ij')
    K1, K2 = cp.meshgrid(p_points, p_points, indexing='ij')
    W_P1, W_P2 = cp.meshgrid(p_weights, p_weights, indexing='ij')
    W_K1, W_K2 = cp.meshgrid(p_weights, p_weights, indexing='ij')

    # Flatten the grids for vectorized computation
    P1_flat = P1.ravel()
    P2_flat = P2.ravel()
    K1_flat = K1.ravel()
    K2_flat = K2.ravel()
    W_P1_flat = W_P1.ravel()
    W_P2_flat = W_P2.ravel()
    W_K1_flat = W_K1.ravel()
    W_K2_flat = W_K2.ravel()

    # For simplicity, consider momenta along x-axis
    p1 = cp.stack([P1_flat, cp.zeros_like(P1_flat), cp.zeros_like(P1_flat)], axis=1)
    p2 = cp.stack([P2_flat, cp.zeros_like(P2_flat), cp.zeros_like(P2_flat)], axis=1)
    k1 = cp.stack([K1_flat, cp.zeros_like(K1_flat), cp.zeros_like(K1_flat)], axis=1)
    k2 = cp.stack([K2_flat, cp.zeros_like(K2_flat), cp.zeros_like(K2_flat)], axis=1)

    # Energy computations
    E_p1 = cp.sqrt(cp.sum(p1 ** 2, axis=1) + m ** 2)
    E_p2 = cp.sqrt(cp.sum(p2 ** 2, axis=1) + m ** 2)
    E_k1 = cp.sqrt(cp.sum(k1 ** 2, axis=1) + m ** 2)
    E_k2 = cp.sqrt(cp.sum(k2 ** 2, axis=1) + m ** 2)

    # Energy conservation delta function (approximated by a narrow Gaussian)
    delta_width = 0.1  # Adjust as needed
    delta_E = cp.exp(- ((E_p1 + E_p2 - E_k1 - E_k2) ** 2) / (2 * delta_width ** 2))
    delta_E /= (delta_width * cp.sqrt(2 * cp.pi))

    # Detector function exponential
    delta_p = p1 + p2 - k1 - k2
    detector_exp = cp.exp(- (sigma2 / 2) * cp.sum(delta_p ** 2, axis=1))

    # Wave packet functions f1 and f2
    def f1(p):
        delta_p = p - p01[None, :]
        exponent = - cp.sum(delta_p ** 2, axis=1) / (2 * Lambda1 ** 2) - 1j * (p @ x01)
        normalization = (1 / (Lambda1 * cp.sqrt(2 * cp.pi))) ** 3
        return normalization * cp.exp(exponent)

    def f2(p):
        delta_p = p - p02[None, :]
        exponent = - cp.sum(delta_p ** 2, axis=1) / (2 * Lambda2 ** 2) - 1j * (p @ x02)
        normalization = (1 / (Lambda2 * cp.sqrt(2 * cp.pi))) ** 3
        return normalization * cp.exp(exponent)

    # Wave packets
    f_p1 = f1(p1)
    f_p2 = f2(p2)
    f_k1 = cp.conj(f1(k1))
    f_k2 = cp.conj(f2(k2))

    # Two-point correlators (simplified to 1 for this example)
    # Adjust according to your model if necessary
    C_p1k1 = 1.0 / (2 * cp.sqrt(cp.sum((p1 - k1) ** 2, axis=1) + m ** 2))
    C_p2k2 = 1.0 / (2 * cp.sqrt(cp.sum((p2 - k2) ** 2, axis=1) + m ** 2))

    # Integrand
    integrand = (
            cp.abs(f_p1 * f_p2 * f_k1 * f_k2) ** 2
            * delta_E * detector_exp * cp.abs(C_p1k1 * C_p2k2) ** 2
    )

    # Weights
    weights = W_P1_flat * W_P2_flat * W_K1_flat * W_K2_flat

    # Compute the integral
    integral = cp.sum(integrand * weights)

    # Multiply by constants
    P = (4 * l / (2 * cp.pi) ** 5) * integral

    # Move result back to CPU memory
    P = cp.asnumpy(P)
    return P


def plot_P_two_particle(
        a_values,
        p01=np.array([1.0, 0.0, 0.0]),
        p02=np.array([-1.0, 0.0, 0.0]),
        sigma2=1.0,
        Lambda1=1.0,
        Lambda2=1.0,
        num_points=30,
        file_out="two_particle_output",
):
    """
    Plot P(D) vs. parameter 'a', where 'a' affects initial positions x01 and x02.

    Parameters:
    - a_values: array-like, values of parameter 'a' to evaluate
    - p01: array-like, initial momentum of particle 1 (default: [1, 0, 0])
    - p02: array-like, initial momentum of particle 2 (default: [-1, 0, 0])
    - sigma2: float, σ^2 parameter of the detector function
    - Lambda1: float, Λ parameter (width) of particle 1's wave packet
    - Lambda2: float, Λ parameter (width) of particle 2's wave packet
    - num_points: int, number of quadrature points for integration
    - file_out: str, output filename for the plot
    """
    P_values = []

    # Loop over 'a' values
    for a in tqdm(a_values, desc="Computing P(D) for different 'a' values"):
        # Set initial positions based on 'a'
        x01 = np.array([a, 0.0, 0.0])
        x02 = np.array([-a, 0.0, 0.0])

        P = compute_P_two_particle(
            p01=p01,
            p02=p02,
            x01=x01,
            x02=x02,
            sigma2=sigma2,
            Lambda1=Lambda1,
            Lambda2=Lambda2,
            num_points=num_points,
        )
        P_values.append(P)

    # Convert P_values to NumPy array
    P_values = np.array(P_values)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(a_values, P_values, marker='o', linestyle='-')
    plt.xlabel(r'Parameter $a$')
    plt.ylabel(r'Probability of Detection $P(D)$')
    plt.title('Probability of Detection vs. Parameter $a$')
    plt.grid(True)
    plt.savefig(os.path.join(os.getenv("HK_FLOW_FILE", ".."), f"{file_out}.png"))
    plt.show()

# Define the range of 'a' values
a_values = np.linspace(0, 5, 10)

# Initial momenta of particles
p01 = np.array([1.0, 0.0, 0.0])
p02 = np.array([-1.0, 0.0, 0.0])

# Call the plotting function
plot_P_two_particle(
    a_values=a_values,
    p01=p01,
    p02=p02,
    sigma2=1.0,
    Lambda1=1.0,
    Lambda2=1.0,
    num_points=30,
    file_out="two_particle_P_vs_a",
)
