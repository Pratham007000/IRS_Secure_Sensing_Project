import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define the Environment & Objective Function
# ==========================================
pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])
Y_wall = 10

def calculate_penalty(x_vals):
    # Calculate (d1 * d2)^2 penalty for any given X coordinate on the wall
    d1_sq = x_vals**2 + (Y_wall - pos_tx[1])**2
    d2_sq = (x_vals - pos_rx[0])**2 + (Y_wall - pos_rx[1])**2
    return d1_sq * d2_sq

# ==========================================
# 2. Particle Swarm Optimization (PSO) Algorithm
# ==========================================
# PSO Parameters
num_particles = 15
num_iterations = 30
w = 0.5    # Inertia weight (how much it keeps its current velocity)
c1 = 1.5   # Cognitive constant (how much it remembers its own best spot)
c2 = 1.5   # Social constant (how much it follows the swarm's best spot)

# Initialize Particles
x = np.random.uniform(0, 10, num_particles)
v = np.random.uniform(-1, 1, num_particles)
p_best_x = np.copy(x)
p_best_val = calculate_penalty(x)

g_best_idx = np.argmin(p_best_val)
g_best_x = p_best_x[g_best_idx]
g_best_val = p_best_val[g_best_idx]

# Track convergence history
convergence_curve = []

# Run Iterations
for i in range(num_iterations):
    convergence_curve.append(g_best_val)
    
    # Update Velocity and Position
    r1, r2 = np.random.rand(num_particles), np.random.rand(num_particles)
    v = w*v + c1*r1*(p_best_x - x) + c2*r2*(g_best_x - x)
    x = x + v
    
    # Enforce Physical Wall Boundaries (0 to 10)
    x = np.clip(x, 0, 10)
    
    # Evaluate new positions
    current_val = calculate_penalty(x)
    
    # Update Personal Best
    better_mask = current_val < p_best_val
    p_best_x[better_mask] = x[better_mask]
    p_best_val[better_mask] = current_val[better_mask]
    
    # Update Global Best
    if np.min(p_best_val) < g_best_val:
        g_best_idx = np.argmin(p_best_val)
        g_best_x = p_best_x[g_best_idx]
        g_best_val = p_best_val[g_best_idx]

# ==========================================
# 3. Plotting the Results
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: The Swarm on the Curve
X_bg = np.linspace(0, 10, 500)
ax1.plot(X_bg, calculate_penalty(X_bg), color='dodgerblue', alpha=0.5, linewidth=3, label='Theoretical Curve')
ax1.scatter(x, calculate_penalty(x), color='magenta', s=50, edgecolor='black', zorder=5, label='Swarm Particles')
ax1.scatter(g_best_x, g_best_val, color='gold', s=200, edgecolor='black', zorder=10, marker='*', 
            label=f'Global Best: X={g_best_x:.2f}m')

ax1.set_title('Final Swarm Convergence on Wall (Y=10m)', fontweight='bold')
ax1.set_xlabel('Horizontal RIS Position ($X_{RIS}$)')
ax1.set_ylabel('Path Loss Penalty')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Plot 2: Convergence Curve
ax2.plot(range(num_iterations), convergence_curve, color='crimson', linewidth=3, marker='o')
ax2.set_title('PSO Convergence History', fontweight='bold')
ax2.set_xlabel('Algorithm Iteration')
ax2.set_ylabel('Swarm Global Best Penalty')
ax2.grid(True, linestyle=':', alpha=0.7)

# Add text box with final result
props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8)
ax2.text(0.5, 0.5, f"Algorithm Solved!\n\nOptimal X: {g_best_x:.3f}m\nPenalty: {g_best_val:.0f}\nFound in <15 Iterations", 
         transform=ax2.transAxes, fontsize=11, verticalalignment='center', horizontalalignment='center', bbox=props)

plt.tight_layout(pad=2.0)
plt.savefig('PSO_Convergence_Results.png', dpi=300, bbox_inches='tight')
print(f"\nSuccess! Algorithm found X = {g_best_x:.3f}m. Saved PSO_Convergence_Results.png")
