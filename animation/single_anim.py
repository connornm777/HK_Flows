import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants and parameters
hbar = 1000.0  # Reduced Planck's constant
c = 1.0  # Speed of light
m = 1.0  # Particle mass
sigma = 1.0  # Width of the initial wave packet
k0 = np.array([5.0, 0.0])  # Initial momentum
r0 = np.array([50.0, 50.0])  # Initial position
rd = np.array([60.0, 50.0])  # Detector position
gamma = 0.1  # Detection rate coefficient

# Spatial grid
Nx, Ny = 100, 100
x = np.linspace(0, 100, Nx)
y = np.linspace(0, 100, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Time parameters
T = 100.0
dt = 0.1
Nt = int(T / dt)

# Courant condition
if dt > dx / c or dt > dy / c:
    raise ValueError("Time step dt is too large for stability.")

# Initialize wavefunction arrays
psi_real = np.zeros((Nx, Ny, Nt))
psi_imag = np.zeros((Nx, Ny, Nt))

# Initial wavefunction
R2 = (X - r0[0]) ** 2 + (Y - r0[1]) ** 2
psi0 = np.exp(-R2 / (2 * sigma ** 2)) * np.cos(k0[0] * (X - r0[0]) + k0[1] * (Y - r0[1]))
psi_real[:, :, 0] = psi0
psi_imag[:, :, 0] = np.exp(-R2 / (2 * sigma ** 2)) * np.sin(k0[0] * (X - r0[0]) + k0[1] * (Y - r0[1]))

# First time derivative (using approximation)
E0 = np.sqrt(k0[0] ** 2 + k0[1] ** 2 + m ** 2)
psi_real[:, :, 1] = psi_real[:, :, 0] + dt * (-E0) * psi_imag[:, :, 0]
psi_imag[:, :, 1] = psi_imag[:, :, 0] + dt * E0 * psi_real[:, :, 0]

# Absorbing boundary layer parameters
absorbing_layer_width = 10
absorbing_coeff = np.linspace(0, 1, absorbing_layer_width)
absorbing_profile = np.ones(Nx)
absorbing_profile[:absorbing_layer_width] = absorbing_coeff
absorbing_profile[-absorbing_layer_width:] = absorbing_coeff[::-1]

# Detection probability array
P_det = np.zeros(Nt)

# Main simulation loop
for n in range(1, Nt - 1):
    # Laplacian (finite differences)
    laplacian_real = (
            (np.roll(psi_real[:, :, n], -1, axis=0) - 2 * psi_real[:, :, n] + np.roll(psi_real[:, :, n], 1,
                                                                                      axis=0)) / dx ** 2 +
            (np.roll(psi_real[:, :, n], -1, axis=1) - 2 * psi_real[:, :, n] + np.roll(psi_real[:, :, n], 1,
                                                                                      axis=1)) / dy ** 2
    )
    laplacian_imag = (
            (np.roll(psi_imag[:, :, n], -1, axis=0) - 2 * psi_imag[:, :, n] + np.roll(psi_imag[:, :, n], 1,
                                                                                      axis=0)) / dx ** 2 +
            (np.roll(psi_imag[:, :, n], -1, axis=1) - 2 * psi_imag[:, :, n] + np.roll(psi_imag[:, :, n], 1,
                                                                                      axis=1)) / dy ** 2
    )

    # Update real and imaginary parts
    psi_real[:, :, n + 1] = (
            2 * psi_real[:, :, n] - psi_real[:, :, n - 1] +
            dt ** 2 * (laplacian_real - m ** 2 * psi_real[:, :, n])
    )
    psi_imag[:, :, n + 1] = (
            2 * psi_imag[:, :, n] - psi_imag[:, :, n - 1] +
            dt ** 2 * (laplacian_imag - m ** 2 * psi_imag[:, :, n])
    )

    # Apply absorbing boundary conditions
    psi_real[:, :, n + 1] *= absorbing_profile[:, np.newaxis]
    psi_imag[:, :, n + 1] *= absorbing_profile[:, np.newaxis]
    psi_real[:, :, n + 1] *= absorbing_profile[np.newaxis, :]
    psi_imag[:, :, n + 1] *= absorbing_profile[np.newaxis, :]

    # Calculate detection probability at this time step
    idx_d = (np.abs(x - rd[0]).argmin(), np.abs(y - rd[1]).argmin())
    prob_density = psi_real[idx_d[0], idx_d[1], n] ** 2 + psi_imag[idx_d[0], idx_d[1], n] ** 2
    P_det[n] = P_det[n - 1] + gamma * prob_density * dt

# Create animation
fig, (ax_wave, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

# Initialize plot elements
extent = [x[0], x[-1], y[0], y[-1]]
im = ax_wave.imshow(psi_real[:, :, 0] ** 2 + psi_imag[:, :, 0] ** 2, extent=extent, origin='lower', cmap='viridis')
ax_wave.set_title('Wavefunction Probability Density')
ax_wave.set_xlabel('x')
ax_wave.set_ylabel('y')
ax_wave.plot(rd[0], rd[1], 'ro', label='Detector')
ax_wave.legend()

bar_container = ax_bar.bar([0], [P_det[0]], width=0.1)
ax_bar.set_xlim(-1, 1)
ax_bar.set_ylim(0, 1.0)
ax_bar.set_xticks([])
ax_bar.set_ylabel('Cumulative Detection Probability')
ax_bar.set_title('Detection Probability Over Time')


def update(frame):
    # Update wavefunction plot
    im.set_array(psi_real[:, :, frame] ** 2 + psi_imag[:, :, frame] ** 2)
    # Update detection probability bar
    bar_container[0].set_height(P_det[frame])
    return im, bar_container


ani = animation.FuncAnimation(fig, update, frames=Nt - 2, interval=50, blit=False)

plt.tight_layout()
plt.show()
