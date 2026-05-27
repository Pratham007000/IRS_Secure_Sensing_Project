import numpy as np
import matplotlib.pyplot as plt

# 1. Exact coordinates from your simulation
pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])
pos_ris = np.array([5, 10])

# 2. Calculate Exact Distances
d1_proj = np.linalg.norm(pos_ris - pos_tx)
d2_proj = np.linalg.norm(pos_rx - pos_ris)
D = d1_proj + d2_proj

# 3. Generate Curve Data keeping D constant
d1_vals = np.linspace(0, D, 1000)
d2_vals = D - d1_vals
penalty = (d1_vals * d2_vals)**2

# Calculations for labels
max_d1 = D / 2
max_penalty = (max_d1 * max_d1)**2
proj_penalty = (d1_proj * d2_proj)**2

# 4. Plot Styling
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the parabolic penalty curve
ax.plot(d1_vals, penalty, color='dodgerblue', linewidth=3, label=f'Cascaded Penalty Curve (Constant $d_1+d_2 = {D:.2f}m$)')

# Maxima (Worst Case)
ax.scatter(max_d1, max_penalty, color='red', s=150, zorder=5)
ax.annotate(f'Absolute Maxima (Worst Signal)\n$d_1={max_d1:.2f}$m, $d_2={max_d1:.2f}$m\nPenalty: {max_penalty:.0f}', 
            xy=(max_d1, max_penalty), xytext=(0, -65), textcoords='offset points', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.9))

# Project coordinates
ax.scatter(d1_proj, proj_penalty, color='lime', s=150, edgecolor='black', zorder=5)
ax.annotate(f'Our Simulation Layout (5, 10)\n$d_1={d1_proj:.2f}$m, $d_2={d2_proj:.2f}$m\nPenalty: {proj_penalty:.0f}', 
            xy=(d1_proj, proj_penalty), xytext=(60, -30), textcoords='offset points', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='green', alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black'))

# Minima
ax.scatter(0, 0, color='purple', s=100, zorder=5)
ax.scatter(D, 0, color='purple', s=100, zorder=5)
ax.annotate(f'Absolute Minima (Best Signal)\n$d_1=0$ or $d_1={D:.2f}$m\nPenalty: 0', 
            xy=(D, 0), xytext=(-100, 30), textcoords='offset points', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='purple', alpha=0.9))

# Labels and grid
ax.set_title(f"Mathematical Proof: Path Loss Penalty vs. RIS Placement", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Distance $d_1$ (Transmitter to RIS) in meters', fontsize=12, fontweight='bold')
ax.set_ylabel('Path Loss Penalty $(d_1 \\times d_2)^2$', fontsize=12, fontweight='bold')
ax.grid(True, linestyle=':', color='gray', alpha=0.6)
ax.set_xlim(-1, D + 1)
ax.set_ylim(-500, max_penalty + 1500)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')

plt.savefig('2D_Penalty_Curve_Simulation.png', dpi=300, bbox_inches='tight')
print(f"\nSuccess! Saved 2D curve as 2D_Penalty_Curve_Simulation.png")
