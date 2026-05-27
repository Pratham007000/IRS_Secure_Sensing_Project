import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. Define General Constraint: d1 + d2 = X
# ==========================================
X_val = 15  # <--- Change this value to ANY total distance!

# Calculate the Mathematical Maxima
d1 = np.linspace(0, X_val, 1000) 
d2 = X_val - d1
penalty = (d1 * d2)**2

max_idx = np.argmax(penalty)
d1_max = d1[max_idx]
d2_max = d2[max_idx]
penalty_max = penalty[max_idx]

print(f"--- For Total Distance (X) = {X_val}m ---")
print(f"Calculated Maxima (Worst Signal):")
print(f"d1 = {d1_max:.2f}m, d2 = {d2_max:.2f}m")
print(f"Penalty = {penalty_max:.0f}")

# ==========================================
# 2. 3D Surface Plot Generation
# ==========================================
grid_max = X_val * 1.2
d1_grid, d2_grid = np.meshgrid(np.linspace(0.1, grid_max, 50), np.linspace(0.1, grid_max, 50))
Z = (d1_grid * d2_grid)**2

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(d1_grid, d2_grid, Z, cmap='magma', alpha=0.7)
ax.plot(d1, d2, penalty, color='cyan', linewidth=4, label=f'Constraint Line: $d_1 + d_2 = {X_val}$m')
ax.scatter(d1_max, d2_max, penalty_max, color='lime', s=150, edgecolor='black', 
           label=f'Maxima: ($d_1$={d1_max:.1f}, $d_2$={d2_max:.1f})\nPenalty = {penalty_max:.0f}', zorder=10)

ax.set_xlabel('Distance $d_1$ (m)')
ax.set_ylabel('Distance $d_2$ (m)')
ax.set_zlabel('Path Loss Penalty $(d_1 \\times d_2)^2$')
ax.set_title(f'3D Cascaded Path Loss Maxima (General $X={X_val}$)', fontweight='bold', pad=20)
ax.legend()

plt.savefig('3D_Maxima_General.png', bbox_inches='tight', dpi=300)
print("\nSuccess! Saved graph as 3D_Maxima_General.png")
