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

def compute_P(p0, x0, theta_i, sigma2=1.0, Lambda=1.0, num_points=30):
    """
    Compute P^{(1)}(D) for given p0, x0, theta_i, sigma2, and Lambda.

    Parameters:
    - p0: float, initial momentum
    - x0: float, initial position away from detector
    - theta_i: float, angle of incidence in radians
    - sigma2: float, σ^2 parameter
    - Lambda: float, Λ parameter
    - num_points: int, number of quadrature points for integration

    Returns:
    - P: float, computed probability
    """
    # Compute prefactor based on Lambda
    prefactor = (1.0+4*sigma2*(Lambda**2)) / (32 * (cp.pi) ** 2 * Lambda ** 4)

    # Integration ranges
    phi_min, phi_max = 0, 2 * cp.pi
    p_min, p_max = 0, 10  # Adjust p_max as needed

    # Use NumPy to compute the Gauss-Legendre quadrature points and weights for u1 and u2
    u_points_np, u_weights_np = np.polynomial.legendre.leggauss(num_points)
    # Convert to CuPy arrays
    u_points = cp.asarray(u_points_np)
    u_weights = cp.asarray(u_weights_np)

    # Trapezoidal rule points for phi1 and phi2 (using CuPy)
    phi_points = cp.linspace(phi_min, phi_max, num_points, endpoint=False)
    phi_weights = (phi_max - phi_min) / num_points

    # Integration points and weights for p (using CuPy)
    p_points = cp.linspace(p_min, p_max, num_points)
    p_weights = cp.zeros_like(p_points)
    p_weights[1:-1] = (p_points[2:] - p_points[:-2]) / 2
    p_weights[0] = p_points[1] - p_points[0]
    p_weights[-1] = p_points[-1] - p_points[-2]

    # Precompute constants
    sin_theta_i = cp.sin(theta_i)
    cos_theta_i = cp.cos(theta_i)
    p0_term = -p0 ** 2 / (2 * Lambda ** 2)

    # Convert scalars to CuPy arrays
    x0 = cp.asarray(x0)

    # Create grids for u1 and u2
    U1, U2 = cp.meshgrid(u_points, u_points, indexing='ij')
    W_U1, W_U2 = cp.meshgrid(u_weights, u_weights, indexing='ij')

    sqrt1 = cp.sqrt(1 - U1 ** 2)
    sqrt2 = cp.sqrt(1 - U2 ** 2)
    delta_u = U2 - U1

    # Grids for phi1 and phi2
    PHI1, PHI2 = cp.meshgrid(phi_points, phi_points, indexing='ij')
    cos_phi_diff_matrix = cp.cos(PHI1 - PHI2)
    cos_phi1 = cp.cos(PHI1)
    cos_phi2 = cp.cos(PHI2)

    # Compute S_matrix and T_matrix
    S_matrix = cos_phi_diff_matrix[None, None, :, :] * sqrt1[:, :, None, None] * sqrt2[:, :, None, None] + U1[:, :, None, None] * U2[:, :, None, None]

    T_matrix = (p0 / (2 * Lambda ** 2)) * (
        sin_theta_i * (cos_phi1[None, None, :, :] * sqrt1[:, :, None, None] + cos_phi2[None, None, :, :] * sqrt2[:, :, None, None]) - cos_theta_i * (U1[:, :, None, None] + U2[:, :, None, None])
    )

    # Im(B) term
    Im_B = x0 * delta_u[:, :, None, None]

    # Initialize P
    P = 0.0

    # Loop over p
    for i_p, (p, w_p) in enumerate(zip(p_points, p_weights)):
        # Compute Re(B)
        Re_B = (
            - (sigma2 + 1 / (2 * Lambda ** 2)) * p ** 2
            + sigma2 * p ** 2 * S_matrix
            + T_matrix * p
            + p0_term
        )

        # Compute the exponential and cosine terms
        exp_Re_B = cp.exp(Re_B)
        cos_Im_B = cp.cos(p * Im_B)

        # Integrand over phi1 and phi2
        integrand_phi = exp_Re_B * cos_Im_B
        integral_phi = phi_weights ** 2 * cp.sum(integrand_phi, axis=(-1, -2))

        # Multiply by p^3 and weights
        integrand = p ** 3 * integral_phi

        # Sum over u1 and u2
        P += cp.sum(W_U1 * W_U2 * w_p * integrand)

    # Multiply by the prefactor
    P = P * prefactor

    # Move result back to CPU memory
    P = cp.asnumpy(P)
    return P

def plot_P_general(p0_values=None, x0_values=None, theta_i_values=None, sigma2_values=[1.0], Lambda_values=[1.0], num_points=30, file_out="output"):
    """
    General plotting function for P^{(1)}(D).

    Parameters:
    - p0_values: array or scalar, values of p0 to use
    - x0_values: array or scalar, values of x0 to use
    - theta_i_values: array or scalar, values of theta_i to use
    - sigma2_values: list or array, values of sigma2 to use (default [1.0])
    - Lambda_values: list or array, values of Lambda to use (default [1.0])
    - num_points: int, number of quadrature points for integration
    """
    # Determine which variables are varying
    variables = {
        'p0': p0_values,
        'x0': x0_values,
        'theta_i': theta_i_values
    }
    varying_vars = {k: v for k, v in variables.items() if isinstance(v, (list, np.ndarray, cp.ndarray))}
    constant_vars = {k: v for k, v in variables.items() if not isinstance(v, (list, np.ndarray, cp.ndarray)) and v is not None}

    # Set default constants if not provided
    defaults = {'p0': 1.0, 'x0': 1.0, 'theta_i': 0.0}
    for var in ['p0', 'x0', 'theta_i']:
        if variables[var] is None:
            variables[var] = defaults[var]
            constant_vars[var] = defaults[var]

    # Ensure sigma2_values and Lambda_values are lists or arrays
    if not isinstance(sigma2_values, (list, np.ndarray)):
        sigma2_values = [sigma2_values]
    if not isinstance(Lambda_values, (list, np.ndarray)):
        Lambda_values = [Lambda_values]

    # Check if sigma2 and Lambda are constants (not being varied)
    sigma2_constant = len(sigma2_values) == 1
    Lambda_constant = len(Lambda_values) == 1

    if len(varying_vars) == 1:
        # Line plot
        var_name, var_values = next(iter(varying_vars.items()))
        var_values = np.array(var_values)
        xlabel_dict = {
            'p0': r'$p_0$',
            'x0': r'$x_0$',
            'theta_i': r'$\theta_i$ (radians)'
        }
        plt.figure(figsize=(8, 6))
        for sigma2 in sigma2_values:
            for Lambda in Lambda_values:
                P_values = []
                label = f"$\sigma^2$={sigma2}, $\Lambda$={Lambda}"
                # Use tqdm for progress bar
                for value in tqdm(var_values, desc=f"Computing P vs {var_name} ({label})"):
                    args = {**constant_vars, var_name: value}
                    P = compute_P(p0=args['p0'], x0=args['x0'], theta_i=args['theta_i'],
                                  sigma2=sigma2, Lambda=Lambda, num_points=num_points)
                    P_values.append(P)
                # Plotting
                plt.plot(var_values, P_values, label=label)
        plt.xlabel(xlabel_dict[var_name])
        plt.ylabel(r'$P^{(1)}(D)$')
        # Title
        constants_str = ', '.join([f"{xlabel_dict[k]} = {v}" for k, v in constant_vars.items()])
        plt.title(f"Probability vs {xlabel_dict[var_name]} ({constants_str})")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.getenv("HK_FLOW_FILE", ".."), f"{file_out}.png"))

    elif len(varying_vars) == 2 and sigma2_constant and Lambda_constant:
        # Heatmap
        var_names = list(varying_vars.keys())
        var_values1 = np.array(varying_vars[var_names[0]])
        var_values2 = np.array(varying_vars[var_names[1]])
        P_values = np.zeros((len(var_values1), len(var_values2)))
        total_values1, total_values2 = len(var_values1), len(var_values2)
        sigma2 = sigma2_values[0]  # Use the constant value
        Lambda = Lambda_values[0]  # Use the constant value

        xlabel_dict = {
            'p0': r'$p_0$',
            'x0': r'$x_0$',
            'theta_i': r'$\theta_i$ (radians)'
        }

        # Use tqdm for progress bar
        for i, value1 in enumerate(tqdm(var_values1, desc=f"Computing P vs {var_names[0]} and {var_names[1]}")):
            for j, value2 in enumerate(var_values2):
                args = {**constant_vars, var_names[0]: value1, var_names[1]: value2}
                P = compute_P(p0=args['p0'], x0=args['x0'], theta_i=args['theta_i'],
                              sigma2=sigma2, Lambda=Lambda, num_points=num_points)
                P_values[i, j] = P
        # Plotting
        plt.figure(figsize=(8, 6))
        X, Y = np.meshgrid(var_values2, var_values1)
        plt.pcolormesh(X, Y, P_values, shading='auto', cmap='viridis')
        plt.colorbar(label=r'$P^{(1)}(D)$')
        # Axis labels with LaTeX
        plt.xlabel(xlabel_dict[var_names[1]])
        plt.ylabel(xlabel_dict[var_names[0]])
        # Title
        constants_str = ', '.join([f"{xlabel_dict[k]} = {v}" for k, v in constant_vars.items()])
        constants_str += f", $\sigma^2$ = {sigma2}, $\Lambda$ = {Lambda}"
        plt.title(f"Probability Heatmap ({constants_str})")
        plt.savefig(os.path.join(os.getenv("HK_FLOW_FILE", ".."), f"{file_out}.png"))

    else:
        print("Error: Please vary one or two variables, and ensure sigma2 and Lambda are constants when plotting a heatmap.")





