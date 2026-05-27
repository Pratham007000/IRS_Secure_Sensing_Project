import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- IEEE Publication Plot Styling ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300
})

# ==========================================
# 1. System Parameters & Physics
# ==========================================
f = 32.8e9              
c = 3e8                 
lambda_ = c / f         

P_tx_dBm = 20           
G_tx_dBi = 15           
G_rx_dBi = 0            
noise_floor_dBm = -90   
P_tx_W = (10 ** (P_tx_dBm / 10)) / 1000  
G_tx_lin = 10 ** (G_tx_dBi / 10)

N = 64                  
alpha = 0.4             
hpbw_deg = 14           
sigma_beam = np.radians(hpbw_deg) / 2.355 

pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])  
pos_ris = np.array([5, 10])

# ==========================================
# 2. Grid Setup & Ray-Casting
# ==========================================
x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)
X, Y = np.meshgrid(x, y)

d_tx = np.sqrt((X - pos_tx[0])**2 + (Y - pos_tx[1])**2)
d_tx[d_tx < 0.1] = 0.1  

Pr_direct_W = P_tx_W * G_tx_lin * (lambda_ / (4 * np.pi * d_tx))**2

slope = (Y - pos_tx[1]) / (X - pos_tx[0] + 1e-9)
slope_min = (1.5 - 2) / 5  
slope_max = (2.5 - 2) / 3  

shadow_mask = (slope >= slope_min) & (slope <= slope_max) & (X >= 3)
Pr_direct_W[shadow_mask] = (10 ** (noise_floor_dBm / 10)) / 1000 

# ==========================================
# 3. RIS Beamforming Model
# ==========================================
d1 = np.linalg.norm(pos_ris - pos_tx)
d2_grid = np.sqrt((X - pos_ris[0])**2 + (Y - pos_ris[1])**2)
d2_grid[d2_grid < 0.1] = 0.1

angle_target = np.arctan2(pos_rx[1] - pos_ris[1], pos_rx[0] - pos_ris[0]) 
angle_grid = np.arctan2(Y - pos_ris[1], X - pos_ris[0])                   
delta_angle = angle_grid - angle_target
beam_gain = np.exp(-(delta_angle**2) / (2 * sigma_beam**2))               

Pr_ris_W = (P_tx_W * G_tx_lin * (lambda_**2) * (N**2) * (alpha**2)) / ((4 * np.pi)**3 * d1**2 * d2_grid**2)
Pr_ris_W = Pr_ris_W * beam_gain 

Pr_total_W = Pr_direct_W + Pr_ris_W

P_dBm_without = 10 * np.log10(Pr_direct_W * 1000 + 1e-15)
P_dBm_with = 10 * np.log10(Pr_total_W * 1000 + 1e-15)

P_dBm_without = np.clip(P_dBm_without, noise_floor_dBm, -40)
P_dBm_with = np.clip(P_dBm_with, noise_floor_dBm, -40)

# Calculate the Difference (Delta Gain in dB)
Delta_Gain_dB = P_dBm_with - P_dBm_without

# ==========================================
# 4. Visualization (Single Large Plot)
# ==========================================
fig, ax = plt.subplots(figsize=(9, 7))
cmap = 'magma' 

ax.set_title("Coverage Enhancement: $\Delta$ RSRP (With vs Without RIS)", pad=15, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("X-Coordinate (m)")
ax.set_ylabel("Y-Coordinate (m)")

ax.grid(True, linestyle=':', color='white', alpha=0.3)
im = ax.imshow(Delta_Gain_dB, extent=[0, 10, 0, 10], origin='lower', cmap=cmap, vmin=0, vmax=40)

levels = [5, 15, 25, 35]
contours = ax.contour(X, Y, Delta_Gain_dB, levels=levels, colors='white', alpha=0.4, linewidths=1)
ax.clabel(contours, inline=True, fontsize=9, fmt='%1.0f dB')

# === MARK D1 and D2 PATHS ===
ax.plot([pos_tx[0], pos_ris[0]], [pos_tx[1], pos_ris[1]], linestyle='--', color='white', linewidth=1.5, alpha=0.7)
ax.plot([pos_ris[0], pos_rx[0]], [pos_ris[1], pos_rx[1]], linestyle='--', color='white', linewidth=1.5, alpha=0.7)

mid_d1 = (pos_tx + pos_ris) / 2
mid_d2 = (pos_ris + pos_rx) / 2
ax.text(mid_d1[0] - 0.5, mid_d1[1] + 0.2, '$d_1$', color='white', fontsize=14, fontweight='bold')
ax.text(mid_d2[0] + 0.2, mid_d2[1] + 0.5, '$d_2$', color='white', fontsize=14, fontweight='bold')
# ============================

ax.plot(pos_tx[0], pos_tx[1], '^', color='cyan', markersize=14, markeredgecolor='black', markeredgewidth=1.2, label='Tx (gNodeB)')
ax.plot(pos_rx[0], pos_rx[1], 'o', color='lawngreen', markersize=12, markeredgecolor='black', markeredgewidth=1.2, label='Rx (Robot)')
ax.plot([4.2, 5.8], [10, 10], '-', color='white', linewidth=5, solid_capstyle='butt', label='8x8 RIS Array')

blocker = patches.Rectangle((3, 1.5), 2, 1, linewidth=1.2, edgecolor='black', facecolor='silver', hatch='////', label='PEC Machinery')
ax.add_patch(blocker)

ax.legend(loc='lower left', framealpha=0.9, edgecolor='black', fancybox=False)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Net Signal Gain ($\Delta$ dB)", weight='bold')

plt.tight_layout(pad=2.0)
plt.savefig('Delta_Gain_Heatmap.png', bbox_inches='tight')
print("\nSuccess! Saved graph as Delta_Gain_Heatmap.png")
