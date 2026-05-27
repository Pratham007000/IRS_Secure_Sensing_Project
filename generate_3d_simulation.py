import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. Exact Coordinates from Simulation
# ==========================================
pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])
pos_ris = np.array([5, 10])

# Calculate Exact Distances
d1_proj = np.linalg.norm(pos_ris - pos_tx)
d2_proj = np.linalg.norm(pos_rx - pos_ris)
D = d1_proj + d2_proj

# Calculate Penalties
max_d1 = D / 2
max_penalty = (max_d1 * max_d1)**2
proj_penalty = (d1_proj * d2_proj)**2

# ==========================================
# 2. 3D Surface Generation
# ==========================================
grid_max = D * 1.2
d1_grid, d2_grid = np.meshgrid(np.linspace(0.1, grid_max, 50), np.linspace(0.1, grid_max, 50))
Z = (d1_grid * d2_grid)**2

# Specific Constraint Line for D = 17.51m
d1_line = np.linspace(0, D, 1000)
d2_line = D - d1_line
Z_line = (d1_line * d2_line)**2

# ==========================================
# 3. Plotting & Styling
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Draw the underlying physics surface
surf = ax.plot_surface(d1_grid, d2_grid, Z, cmap='magma', alpha=0.5)

# Draw the 17.51m Constraint Line
ax.plot(d1_line, d2_line, Z_line, color='cyan', linewidth=5, label=f'Constraint Line: $d_1 + d_2 = {D:.2f}$m', zorder=4)

# Mark the Absolute Maxima (Red)
ax.scatter(max_d1, max_d1, max_penalty, color='red', s=200, edgecolor='white', 
           label=f'Absolute Maxima (Worst Signal)\nPenalty: {max_penalty:.0f}', zorder=5)

# Mark the Actual Simulation Position (Lime Green)
ax.scatter(d1_proj, d2_proj, proj_penalty, color='lime', s=200, edgecolor='black', 
           label=f'Our RIS Layout (5, 10)\nPenalty: {proj_penalty:.0f}', zorder=6)

# Labels and Formatting
ax.set_xlabel('Distance $d_1$ (m)', fontweight='bold')
ax.set_ylabel('Distance $d_2$ (m)', fontweight='bold')
ax.set_zlabel('Path Loss Penalty $(d_1 \\times d_2)^2$', fontweight='bold')
ax.set_title(f'3D Simulation Reality (Fixed Total Distance = {D:.2f}m)', fontweight='bold', fontsize=14, pad=20)

# Custom Legend to stand out
ax.legend(loc='upper left', framealpha=0.9, edgecolor='black', fontsize=11)

# Rotate the 3D camera to get the best view of both dots
ax.view_init(elev=25, azim=-55)

plt.savefig('3D_Simulation_Reality.png', bbox_inches='tight', dpi=300)
print(f"\nSuccess! Saved customized 3D graph as 3D_Simulation_Reality.png")
