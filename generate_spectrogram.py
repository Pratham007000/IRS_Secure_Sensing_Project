import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. System Parameters & Geometry
# ==========================================
f = 32.8e9              # Carrier Frequency: 32.8 GHz
c = 3e8                 # Speed of light
lambda_ = c / f         # Wavelength (approx 9.14 mm)

# Hardware Params
P_tx_dBm = 20           # Transmit Power: 20 dBm (100 mW)
G_tx_dBi = 15           # gNodeB Horn Antenna Gain
G_rx_dBi = 0            # Robot Omni Antenna Gain
N = 64                  # RIS Elements (8x8)
alpha = 0.4             # RIS Amplitude Reflection (ON State from Skyworks)
noise_floor_dBm = -90   # Assumed Thermal Noise Floor

# Coordinates (meters)
pos_tx = np.array([0, 2])
pos_rx = np.array([8, 2.5])
pos_ris = np.array([5, 10])

# ==========================================
# 2. Mathematical Model (Link Budget)
# ==========================================
# Calculate Distances
d_direct = np.linalg.norm(pos_rx - pos_tx)
d1 = np.linalg.norm(pos_ris - pos_tx)
d2 = np.linalg.norm(pos_rx - pos_ris)

# State 0: Without RIS (Blockage)
P_rx_without_RIS_dBm = noise_floor_dBm 

# State 1: With RIS (Cascaded Path)
P_tx_linear = 10 ** (P_tx_dBm / 10)
G_tx_linear = 10 ** (G_tx_dBi / 10)
G_rx_linear = 10 ** (G_rx_dBi / 10)

# Cascaded channel gain formula
cascaded_gain = ( (G_tx_linear * G_rx_linear * (lambda_**2) * (N**2) * (alpha**2)) / 
                  ((4 * np.pi)**3 * (d1**2) * (d2**2)) )

P_rx_with_RIS_linear = P_tx_linear * cascaded_gain
P_rx_with_RIS_dBm = 10 * np.log10(P_rx_with_RIS_linear)

print(f"--- Power Calculations ---")
print(f"Distance Tx -> RIS (d1): {d1:.2f} m")
print(f"Distance RIS -> Rx (d2): {d2:.2f} m")
print(f"Received Power (Without RIS): {P_rx_without_RIS_dBm:.2f} dBm")
print(f"Received Power (With RIS):    {P_rx_with_RIS_dBm:.2f} dBm\n")

# ==========================================
# 3. Signal Generation (Time Domain)
# ==========================================
fs = 10000              # Sampling frequency (10 kHz for baseband simulation)
t = np.arange(0, 1.0, 1/fs) # 1 second duration
f_tone = 2000           # CW Pilot Tone at 2 kHz (baseband offset)

# Generate Noise
noise_power = 10 ** (noise_floor_dBm / 10)
voltage_noise = np.sqrt(noise_power) * np.random.randn(len(t))

# Signal Without RIS (Just Noise)
sig_without_ris = voltage_noise

# Signal With RIS (Noise + Pilot Tone)
voltage_signal = np.sqrt(P_rx_with_RIS_linear)
sig_with_ris = voltage_noise + voltage_signal * np.cos(2 * np.pi * f_tone * t)

# ==========================================
# 4. Spectrogram Visualization
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Without RIS
Pxx1, freqs1, bins1, im1 = ax1.specgram(sig_without_ris, NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
ax1.set_title("Spectrogram: Without RIS (Blocked)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_ylim(0, 5000)
fig.colorbar(im1, ax=ax1, label='Power/Frequency (dB/Hz)')

# Plot 2: With RIS
Pxx2, freqs2, bins2, im2 = ax2.specgram(sig_with_ris, NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
ax2.set_title("Spectrogram: With RIS (vLoS Active)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_ylim(0, 5000)
fig.colorbar(im2, ax=ax2, label='Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()
