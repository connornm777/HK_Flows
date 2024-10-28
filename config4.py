from calcutils import plot_P_general
import numpy as np

num_points = 40
n = 20
x0_vals = np.linspace(0, 10., n)
theta_i_vals = 0
p0_vals = np.linspace(0, 10., n)
file_out = "pvxp"
sigma2_vals = [0.0]
Lambda_vals = [1.0]

# Compute and plot P vs x0 for different sigma2 and Lambda values (Line Plot)
plot_P_general(p0_values=p0_vals, x0_values=x0_vals, theta_i_values=theta_i_vals,
               sigma2_values=sigma2_vals, Lambda_values=Lambda_vals, num_points=num_points, file_out=file_out)