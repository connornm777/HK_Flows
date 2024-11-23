import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
# Enable LaTeX rendering in Matplotlib
plt.rcParams['text.usetex'] = True

# Constants and parameters
l = 1000.0  # Given parameter
m = 10.0  # Mass
sigma = 1.0
Lambda = 1.0
p0 = np.array([5.0, 0.0, 0.0])  # Initial momentum
x0 = np.array([-5.0, 0.0, 0.0])  # Initial position
x_n_list = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]  # x_n for n=1,2


# Define time-dependent alpha_n(t)
def alpha_n(t, n):
    x_n = x_n_list[n - 1]

    A = sigma ** 2 + 1 / (4 * Lambda ** 2)
    C = 1 / (2 * Lambda ** 2) + (t ** 2) / (2 * m ** 2 * A)
    D = (p0 / Lambda ** 2) - ((t / (m * A)) * (x0 - x_n))

    # Calculate exponentials
    D_squared = np.dot(D, D)
    exponent = (D_squared) / (4 * C) - (np.dot(p0, p0)) / (2 * Lambda ** 2) - (np.dot(x0 - x_n, x0 - x_n)) / (2 * A)

    # Calculate prefactors
    prefactor = (l / (8 * Lambda ** 3 * m)) * (1 / (np.pi ** (9 / 4) * A ** (3 / 2) * C ** (3 / 4)))

    alpha = prefactor * np.exp(exponent)
    return alpha


# Define the time span and evaluation points
T = 20  # Final time
N = 1000  # Number of time points
t_eval = np.linspace(0, T, N)


# Define the function to solve the ODE system for given parameters
def run_simulation(P0):
    # Define the system of differential equations with time-dependent alphas
    def dP_dt(t, P):
        P1, P2, P3, P4 = P
        a1 = alpha_n(t, 1)
        a2 = alpha_n(t, 2)
        dP1_dt = - (a1 + a2) * P1
        dP2_dt = a1 * P1 - a2 * P2
        dP3_dt = - a1 * P3 + a2 * P1
        dP4_dt = a1 * P3 + a2 * P2
        return [dP1_dt, dP2_dt, dP3_dt, dP4_dt]

    # Solve the ODE system
    solution = solve_ivp(dP_dt, [0, T], P0, t_eval=t_eval)
    return solution


# Initial conditions
P0_values = [[1.0, 0.0, 0.0, 0.0]]  # Initial probabilities
file_out = 'inarow'

# Run simulations
if __name__ == '__main__':
    with Pool() as pool:
        results = list(tqdm(pool.imap(run_simulation, P0_values), total=len(P0_values)))

    # Plot the results
    for idx, solution in enumerate(results):
        plt.figure(figsize=(10, 6))
        plt.plot(solution.t, solution.y[0], label='$P(\\bar{D}_1, \\bar{D}_2)$')
        plt.plot(solution.t, solution.y[1], label='$P(D_1, \\bar{D}_2)$')
        plt.plot(solution.t, solution.y[2], label='$P(\\bar{D}_1, D_2)$')
        plt.plot(solution.t, solution.y[3], label='$P(D_1, D_2)$')
        plt.title(f'Simulation {idx + 1}')
        plt.xlabel('Time t')
        plt.ylabel('Probabilities P')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(os.getenv("SEMINAR_FILE", ".."), f"{file_out}.png"))
