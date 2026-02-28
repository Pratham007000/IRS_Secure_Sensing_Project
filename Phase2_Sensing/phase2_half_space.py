import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
#  PHASE 2 FINAL: HALF-SPACE SENSING (The Physics Fix)
#  Constraint: 1-Bit Real Weights cannot distinguish Left/Right.
#  Solution: Restrict sensing to 0 to 60 degrees (One side of the room).
# =============================================================================

def run_half_space_solution():
    print("--- STEP 1: Generating Codebook ---")
    N_elements = 64
    Num_Beams = 64
    
    # We use DFT because it creates sharp peaks
    dft_matrix = np.fft.fft(np.eye(N_elements))
    indices = np.linspace(0, N_elements, Num_Beams, endpoint=False, dtype=int)
    ideal_codebook = dft_matrix[:, indices]
    
    # 1-Bit Quantization + Amplitude Dip
    phases = np.angle(ideal_codebook) % (2 * np.pi)
    w_phases = np.where((phases > np.pi/2) & (phases < 3*np.pi/2), np.pi, 0.0)
    w_amps = np.where(w_phases > 1.0, 0.4, 0.9)
    W_complex = w_amps * np.exp(1j * w_phases)

    print("--- STEP 2: Simulating 5,000 Users (0 to 60 deg ONLY) ---")
    num_samples = 5000
    X_raw = np.zeros((num_samples, Num_Beams))
    Y_labels = np.zeros((num_samples,))
    
    # Physics Constants
    wavelength = 3e8 / 32.8e9
    d = wavelength / 2
    n_idx = np.arange(N_elements)
    
    for i in range(num_samples):
        # *** THE FIX: RESTRICT TO POSITIVE ANGLES ***
        angle_deg = np.random.uniform(0, 60) 
        dist = np.random.uniform(2, 10)
        Y_labels[i] = angle_deg
        
        # Channel
        angle_rad = np.deg2rad(angle_deg)
        spatial_freq = (2 * np.pi * d / wavelength) * np.sin(angle_rad)
        h = np.exp(1j * n_idx * spatial_freq) * (1/dist)
        
        # Sensing
        signal = np.abs(h @ W_complex)**2
        
        # Add Noise
        noise = np.random.normal(0, 1e-10, Num_Beams)
        X_raw[i, :] = signal + np.abs(noise)

    print("--- STEP 3: Normalization ---")
    # Log Scale + Row Mean Subtraction
    X_dB = 10 * np.log10(X_raw + 1e-15)
    row_means = np.mean(X_dB, axis=1, keepdims=True)
    X_norm = X_dB - row_means

    print("--- STEP 4: Training Random Forest ---")
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_labels, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, Y_train)
    
    print("--- STEP 5: Evaluation ---")
    preds = model.predict(X_test)
    error = mean_absolute_error(Y_test, preds)
    
    print("="*50)
    print(f"FINAL ACCURACY: Mean Absolute Error = {error:.2f} degrees")
    print("="*50)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:50], 'bo', label='True Angle')
    plt.plot(preds[:50], 'rx', label='Prediction')
    plt.title(f"Half-Space Sensing Accuracy (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_half_space_solution()
