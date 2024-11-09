import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Parameters
x0 = 5.0            # Initial z-position of the wave-packet
p0 = 1.0            # Magnitude of the average momentum
theta_i = np.pi / 6 # Incident angle in radians (30 degrees)
sigma_x = 1.0       # Width of particle
sigma_d = 0.1       # Width of detector

# Define the trajectory of the wave-packet
def x_mean(z):
    return np.tan(theta_i) * (x0 - z)

# Determine the full range for x and z to ensure Gaussians are not cut off
# Set equal ranges for x and z axes
x_min = -5.0
x_max = 5.0
z_min = -2.0
z_max = 8.0

# Create x and z values
num_points = 500
x_vals = np.linspace(x_min, x_max, num_points)
z_vals = np.linspace(z_min, z_max, num_points)
X, Z = np.meshgrid(x_vals, z_vals)

# Calculate the Gaussian amplitude at each point for the particle
Psi_particle = np.exp(- X**2 / (2 * sigma_x**2)) * np.exp(-(Z-x0)**2/(2*sigma_x**2))

# Calculate the Gaussian amplitude for the detector at the origin
Psi_detector = np.exp(- X**2 / (2 * sigma_d**2)) * np.exp(- Z**2 / (2 * sigma_d**2))

# Total amplitude is the sum of the particle and detector Gaussians
Psi_total = Psi_particle + Psi_detector

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')

# Plot the total Gaussian amplitude
surf_total = ax.plot_surface(X, Z, Psi_total, cmap='viridis', alpha=0.8, edgecolor='none')

# Plot the initial position of the particle
particle_center_x = x_mean(x0)
particle_center_z = x0
particle_center_amp = np.exp(0)

ax.plot([particle_center_x], [particle_center_z], [particle_center_amp], 'ro', markersize=8)

# Plot the detector's center at the origin
detector_center_x = 0
detector_center_z = 0
detector_center_amp = np.exp(0)

ax.plot([detector_center_x], [detector_center_z], [detector_center_amp], 'bo', markersize=8)

# Plot the momentum vector in the x-z plane
p0_x = p0 * np.sin(theta_i)
p0_z = -p0 * np.cos(theta_i)
ax.quiver3D(particle_center_x, particle_center_z, particle_center_amp,
          p0_x, p0_z, 0, length=1.0, arrow_length_ratio=0.1, color='red', label='p0')

# Plot an arrow from the detector to the particle
delta_x = particle_center_x - detector_center_x
delta_z = particle_center_z - detector_center_z
ax.quiver3D(detector_center_x, detector_center_z, detector_center_amp,
          delta_x, delta_z, 0, length=1.0, arrow_length_ratio=0.1, color='green', label='x0')

# Label the angle of incidence
# Draw an arc to represent the angle
angle_arc = np.linspace(-np.pi/2, -np.pi/2 + theta_i, 100)
arc_radius = 1.0
arc_x = particle_center_x + arc_radius * np.cos(angle_arc)
arc_z = particle_center_z + arc_radius * np.sin(angle_arc)
arc_y = np.zeros_like(arc_x) + particle_center_amp
ax.plot(arc_x, arc_z, arc_y, color='red')

# Place the angle label
angle_label_x = particle_center_x + (arc_radius + 0.2) * np.cos(-np.pi/2 + theta_i/2)
angle_label_z = particle_center_z + (arc_radius + 0.2) * np.sin(-np.pi/2 + theta_i/2)
ax.text(angle_label_x, angle_label_z, particle_center_amp + 0.1, r'$\theta_i$', color='red')

# Label the centers of the Gaussians
ax.text(particle_center_x, particle_center_z, particle_center_amp + 0.2, 'Particle Center', color='red', ha='center')
ax.text(detector_center_x, detector_center_z, detector_center_amp + 0.2, 'Detector Center', color='blue', ha='center')

# Remove tick mark number labels
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Remove the vertical axis label
ax.set_zlabel('')

# Set equal limits for x and z axes
#ax.set_xlim(x_min, x_max)
#ax.set_ylim(z_min, z_max)

# Set the same tick marks for x and z axes
#tick_vals = np.linspace(x_min, x_max, 5)
#ax.set_xticks(tick_vals)
#ax.set_yticks(tick_vals)

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('Visualization of the Wave-Packet and Detector')

# Customize the view angle for better visualization
ax.view_init(elev=1000, azim=-30)

# Adjust the aspect ratio manually
ax.set_box_aspect([1,1,0.5])  # x:y:z ratios

plt.show()
# Show the plot
plt.savefig(os.path.join(os.getenv("HK_FLOW_FILE"), 'wave_plot.jpg'))
