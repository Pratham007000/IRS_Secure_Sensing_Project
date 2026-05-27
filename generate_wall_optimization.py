import numpy as np
import matplotlib.pyplot as plt

# 1. Room Coordinates
pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])
Y_wall = 10

# 2. Slide the RIS horizontally from X=0 to X=10
X_vals = np.linspace(0, 10, 1000)

# Calculate d1 and d2 for every point on the wall
d1_sq = X_vals**2 + (Y_wall - pos_tx[1])**2
d2_sq = (X_vals - pos_rx[0])**2 + (Y_wall - pos_rx[1])**2

# Cascaded Penalty (d1 * d2)^2
penalty = d1_sq * d2_sq

# 3. Find the Mathematical Optimum on the wall
min_idx = np.argmin(penalty)
X_opt = X_vals[min_idx]
opt_penalty = penalty[min_idx]

# 4. Calculate your Simulation Point (X=5)
X_sim = 5.0
d1_sim_sq = X_sim**2 + (Y_wall - pos_tx[1])**2
d2_sim_sq = (X_sim - pos_rx[0])**2 + (Y_wall - pos_rx[1])**2
sim_penalty = d1_sim_sq * d2_sim_sq

# ==========================================
# 5. Plotting the Constrained Reality
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the curve
ax.plot(X_vals, penalty, color='mediumblue', linewidth=3, label=f'Wall-Constrained Penalty Curve ($Y={Y_wall}$m)')

# Mark the True Optimum
ax.scatter(X_opt, opt_penalty, color='gold', s=180, edgecolor='black', zorder=5,
           label=f'Optimal Wall Position\n$X={X_opt:.2f}$m (Penalty: {opt_penalty:.0f})')

# Mark the Simulation Choice
ax.scatter(X_sim, sim_penalty, color='lime', s=150, edgecolor='black', zorder=5,
           label=f'Our Deployment\n$X={X_sim:.2f}$m (Penalty: {sim_penalty:.0f})')

# Draw vertical lines to the X-axis for clarity
ax.vlines(x=X_opt, ymin=0, ymax=opt_penalty, color='gold', linestyle='--', alpha=0.7)
ax.vlines(x=X_sim, ymin=0, ymax=sim_penalty, color='lime', linestyle='--', alpha=0.7)

# Formatting
ax.set_title(f"Optimization Under Physical Constraints (Fixed Wall at Y={Y_wall}m)", fontweight='bold', pad=15)
ax.set_xlabel('Horizontal Position on Wall ($X_{RIS}$) in meters', fontweight='bold')
ax.set_ylabel('Cascaded Path Loss Penalty', fontweight='bold')
ax.grid(True, linestyle=':', color='gray', alpha=0.5)
ax.set_xlim(0, 10)
ax.set_ylim(4000, 10500)
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')

# Add a text box explaining the negligible difference
efficiency = (opt_penalty / sim_penalty) * 100
props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8, edgecolor='silver')
ax.text(0.5, 9500, f"Analysis:\nThe mathematical optimum is at X=4.35m.\nOur deployment at X=5.0m achieves\n{efficiency:.1f}% of the maximum theoretical\nperformance possible on this wall.", 
        fontsize=10, bbox=props, family='monospace')

plt.savefig('Wall_Constrained_Optimization.png', dpi=300, bbox_inches='tight')
print(f"\nSuccess! Saved graph as Wall_Constrained_Optimization.png")
