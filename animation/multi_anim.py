import numpy as np
import matplotlib.pyplot as plt
import os

# Constants and parameters
hbar = 100.0  # Reduced Planck's constant
c = 1.0        # Speed of light
m = 1.0        # Particle mass
sigma = 1.0    # Width of the initial wave packet
k0 = np.array([5.0, 0.0])  # Initial momentum
r0 = np.array([50.0, 50.0])  # Initial position
rd1 = np.array([60.0, 50.0])  # Detector 1 position
rd2 = np.array([80.0, 50.0])  # Detector 2 position
gamma1 = 1.0   # Detection rate coefficient for detector 1
gamma2 = 1.0   # Detection rate coefficient for detector 2

# Spatial grid
Nx, Ny = 100, 100
x = np.linspace(0, 100, Nx)
y = np.linspace(0, 100, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Time parameters
T = 60.0
dt = 0.4
Nt = int(T / dt)

# Courant condition
if dt > dx / c or dt > dy / c:
    raise ValueError("Time step dt is too large for stability.")

# Initialize wavefunction arrays
psi_real = np.zeros((Nx, Ny, Nt))
psi_imag = np.zeros((Nx, Ny, Nt))

# Initial wavefunction
R2 = (X - r0[0])**2 + (Y - r0[1])**2
phase = k0[0] * (X - r0[0]) + k0[1] * (Y - r0[1])
psi0 = np.exp(-R2 / (2 * sigma**2))
psi_real[:, :, 0] = psi0 * np.cos(phase)
psi_imag[:, :, 0] = psi0 * np.sin(phase)

# First time derivative (using approximation)
E0 = np.sqrt(k0[0]**2 + k0[1]**2 + m**2)
psi_real[:, :, 1] = psi_real[:, :, 0] + dt * (-E0) * psi_imag[:, :, 0]
psi_imag[:, :, 1] = psi_imag[:, :, 0] + dt * E0 * psi_real[:, :, 0]

# Absorbing boundary layer parameters
absorbing_layer_width = 10
absorbing_coeff = np.linspace(1, 0, absorbing_layer_width)
absorbing_profile = np.ones(Nx)
absorbing_profile[:absorbing_layer_width] = absorbing_coeff
absorbing_profile[-absorbing_layer_width:] = absorbing_coeff
absorbing_profile_2d = np.outer(absorbing_profile, absorbing_profile)

# Detection probability arrays
P_D1 = np.zeros(Nt)  # Cumulative detection probability for detector 1
P_D2 = np.zeros(Nt)  # Cumulative detection probability for detector 2

# Main simulation loop
for n in range(1, Nt - 1):
    # Laplacian (finite differences)
    laplacian_real = (
        (np.roll(psi_real[:, :, n], -1, axis=0) - 2 * psi_real[:, :, n] + np.roll(psi_real[:, :, n], 1, axis=0)) / dx**2 +
        (np.roll(psi_real[:, :, n], -1, axis=1) - 2 * psi_real[:, :, n] + np.roll(psi_real[:, :, n], 1, axis=1)) / dy**2
    )
    laplacian_imag = (
        (np.roll(psi_imag[:, :, n], -1, axis=0) - 2 * psi_imag[:, :, n] + np.roll(psi_imag[:, :, n], 1, axis=0)) / dx**2 +
        (np.roll(psi_imag[:, :, n], -1, axis=1) - 2 * psi_imag[:, :, n] + np.roll(psi_imag[:, :, n], 1, axis=1)) / dy**2
    )

    # Update real and imaginary parts
    psi_real[:, :, n + 1] = (
        2 * psi_real[:, :, n] - psi_real[:, :, n - 1] +
        dt**2 * (laplacian_real - m**2 * psi_real[:, :, n])
    )
    psi_imag[:, :, n + 1] = (
        2 * psi_imag[:, :, n] - psi_imag[:, :, n - 1] +
        dt**2 * (laplacian_imag - m**2 * psi_imag[:, :, n])
    )

    # Apply absorbing boundary conditions
    psi_real[:, :, n + 1] *= absorbing_profile_2d
    psi_imag[:, :, n + 1] *= absorbing_profile_2d

    # Calculate detection probabilities at this time step
    idx_d1_x = np.abs(x - rd1[0]).argmin()
    idx_d1_y = np.abs(y - rd1[1]).argmin()
    prob_density_d1 = psi_real[idx_d1_x, idx_d1_y, n]**2 + psi_imag[idx_d1_x, idx_d1_y, n]**2

    idx_d2_x = np.abs(x - rd2[0]).argmin()
    idx_d2_y = np.abs(y - rd2[1]).argmin()
    prob_density_d2 = psi_real[idx_d2_x, idx_d2_y, n]**2 + psi_imag[idx_d2_x, idx_d2_y, n]**2

    # Update cumulative detection probabilities
    P_D1[n] = P_D1[n - 1] + gamma1 * prob_density_d1 * dt
    P_D2[n] = P_D2[n - 1] + gamma2 * prob_density_d2 * dt

# Ensure probabilities do not exceed 1
P_D1 = np.clip(P_D1, 0, 1)
P_D2 = np.clip(P_D2, 0, 1)

# Calculate joint probabilities
P_D1_D2 = P_D1 * P_D2
P_D1_notD2 = P_D1 * (1 - P_D2)
P_notD1_D2 = (1 - P_D1) * P_D2
P_notD1_notD2 = (1 - P_D1) * (1 - P_D2)

# Create the directory if it doesn't exist
output_dir = '/home/connor/Dropbox/Research/Writing/SeminarTues/animation/'
os.makedirs(output_dir, exist_ok=True)

# Create figure and axes
fig, (ax_wave, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))


# Initialize plot elements
extent = [x[0], x[-1], y[0], y[-1]]
im = ax_wave.imshow(psi_real[:, :, 0]**2 + psi_imag[:, :, 0]**2, extent=extent, origin='lower', cmap='viridis')
ax_wave.set_title('Wavefunction Probability Density')
ax_wave.set_xlabel('x')
ax_wave.set_ylabel('y')
ax_wave.plot(rd1[0], rd1[1], 'ro', label='Detector 1')
ax_wave.plot(rd2[0], rd2[1], 'go', label='Detector 2')
ax_wave.legend()

# Bar chart setup
probabilities = [P_D1_D2[0], P_D1_notD2[0], P_notD1_D2[0], P_notD1_notD2[0]]
x_positions = np.arange(len(probabilities))
labels = ['$P(D_1, D_2)$', '$P(D_1, \\bar{D}_2)$', '$P(\\bar{D}_1, D_2)$', '$P(\\bar{D}_1, \\bar{D}_2)$']
colors = ['purple', 'red', 'green', 'blue']
bars = ax_bar.bar(x_positions, probabilities, color=colors, edgecolor='black')
ax_bar.set_xticks(x_positions)
ax_bar.set_xticklabels(labels)
ax_bar.set_ylim(0, 1)
ax_bar.set_ylabel('Probability')
ax_bar.set_title('Detection Probabilities')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(4)]
ax_bar.legend(handles=legend_elements, loc='upper right')

def update(frame):
    # Update wavefunction plot
    im.set_array(psi_real[:, :, frame]**2 + psi_imag[:, :, frame]**2)

    # Update probabilities
    probabilities = [P_D1_D2[frame], P_D1_notD2[frame], P_notD1_D2[frame], P_notD1_notD2[frame]]
    cumulative = np.cumsum(probabilities)
    # Update bar segments
    for i, bar in enumerate(bars):
        bar.set_height(probabilities[i])
        bar.set_color(colors[i])

    # Save the current frame
    filename = os.path.join(output_dir, f'frame-{frame:02d}.png')
    plt.savefig(filename)
    print(f'Saved {filename}')

# Instead of using FuncAnimation, manually update and save frames
for frame in range(Nt - 2):
    update(frame)

plt.close(fig)
