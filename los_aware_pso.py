import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define the Environment & Constraints
# ==========================================
pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])
Y_wall = 10

# The PEC Machinery Bounding Box: [X_min, X_max, Y_min, Y_max]
blocker = np.array([3.0, 5.0, 1.5, 2.5])

def is_blocked(start_pos, end_positions, box):
    """Fires 100 sample points along the ray to check for blocker intersection"""
    t = np.linspace(0, 1, 100)[:, None, None] 
    rays = start_pos + t * (end_positions - start_pos)
    
    # Check if any points fall inside the blocker box
    hit_x = (rays[:, :, 0] >= box[0]) & (rays[:, :, 0] <= box[1])
    hit_y = (rays[:, :, 1] >= box[2]) & (rays[:, :, 1] <= box[3])
    
    # Return True if a ray hit the box at ANY point
    return np.any(hit_x & hit_y, axis=0)

def calculate_penalty(x_vals):
    # 1. Calculate Standard Distance Penalty
    d1_sq = x_vals**2 + (Y_wall - pos_tx[1])**2
    d2_sq = (x_vals - pos_rx[0])**2 + (Y_wall - pos_rx[1])**2
    penalty = d1_sq * d2_sq
    
    # 2. THE GEOMETRIC CONSTRAINT: Line of Sight Check
    ris_positions = np.column_stack((x_vals, np.full_like(x_vals, Y_wall)))
    tx_blocked = is_blocked(pos_tx, ris_positions, blocker)
    rx_blocked = is_blocked(pos_rx, ris_positions, blocker)
    
    # 3. Apply the "Death Penalty"
    penalty[tx_blocked | rx_blocked] = 1e9  # 1 Billion Penalty
    return penalty

# ==========================================
# 2. Particle Swarm Optimization (PSO)
# ==========================================
num_particles = 15
num_iterations = 30
w, c1, c2 = 0.5, 1.5, 1.5

# Initialize Swarm
x = np.random.uniform(0, 10, num_particles)
v = np.random.uniform(-1, 1, num_particles)
p_best_x = np.copy(x)
p_best_val = calculate_penalty(x)

g_best_idx = np.argmin(p_best_val)
g_best_x = p_best_x[g_best_idx]
g_best_val = p_best_val[g_best_idx]
convergence_curve = []

# Run Iterations
for i in range(num_iterations):
    convergence_curve.append(g_best_val)
    
    r1, r2 = np.random.rand(num_particles), np.random.rand(num_particles)
    v = w*v + c1*r1*(p_best_x - x) + c2*r2*(g_best_x - x)
    x = np.clip(x + v, 0, 10) # Enforce Room Boundaries
    
    current_val = calculate_penalty(x)
    
    better_mask = current_val < p_best_val
    p_best_x[better_mask] = x[better_mask]
    p_best_val[better_mask] = current_val[better_mask]
    
    if np.min(p_best_val) < g_best_val:
        g_best_idx = np.argmin(p_best_val)
        g_best_x = p_best_x[g_best_idx]
        g_best_val = p_best_val[g_best_idx]

# ==========================================
# 3. Plotting Results
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

X_bg = np.linspace(0, 10, 500)
bg_penalty = calculate_penalty(X_bg)

# Handle the graph scale so the 1 Billion penalty doesn't ruin the image
bg_penalty[bg_penalty > 20000] = np.nan 

ax1.plot(X_bg, bg_penalty, color='dodgerblue', alpha=0.5, linewidth=3, label='LoS Clear Curve')
ax1.scatter(x, calculate_penalty(x), color='magenta', s=50, edgecolor='black', zorder=5, label='Swarm Particles')
ax1.scatter(g_best_x, g_best_val, color='gold', s=200, edgecolor='black', zorder=10, marker='*', label=f'Global Best: X={g_best_x:.2f}m')

ax1.set_title('LoS-Aware PSO Convergence Profile', fontweight='bold')
ax1.set_xlabel('Horizontal RIS Position ($X_{RIS}$)')
ax1.set_ylabel('Path Loss Penalty')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

ax2.plot(range(num_iterations), convergence_curve, color='crimson', linewidth=3, marker='o')
ax2.set_title('PSO Convergence History', fontweight='bold')
ax2.set_xlabel('Algorithm Iteration')
ax2.set_ylabel('Swarm Global Best Penalty')
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig('LoS_Aware_PSO.png', dpi=300, bbox_inches='tight')
print(f"\nSuccess! LoS-Aware Algorithm found X = {g_best_x:.3f}m.")
