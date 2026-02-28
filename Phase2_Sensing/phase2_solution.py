import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
#  PHASE 2 FINAL SOLUTION: DFT BEAMS + DISTANCE REMOVAL + RANDOM FOREST
# =============================================================================

def run_solution():
    print("--- STEP 1: Generating DFT Codebook (The Physics-Proven Choice) ---")
    N_elements = 64
    Num_Beams = 64
    
    # We use the DFT matrix because your diagnostic PROVED it creates strong peaks.
    # Even though it has 'ghosts', the ghosts move differently for every beam.
    # The AI will use this 'difference' to find the true angle.
    dft_matrix = np.fft.fft(np.eye(N_elements))
    indices = np.linspace(0, N_elements, Num_Beams, endpoint=False, dtype=int)
    ideal_codebook = dft_matrix[:, indices]
    
    # 1-Bit Quantization
    phases = np.angle(ideal_codebook) % (2 * np.pi)
    w_phases = np.where((phases > np.pi/2) & (phases < 3*np.pi/2), np.pi, 0.0)
    
    # Hardware Amplitude Dip
    w_amps = np.where(w_phases > 1.0, 0.4, 0.9)
    W_complex = w_amps * np.exp(1j * w_phases)

    print("--- STEP 2: Simulating 5,000 Users ---")
    num_samples = 5000
    X_raw = np.zeros((num_samples, Num_Beams))
    Y_labels = np.zeros((num_samples,))
    
    # Physics Constants
    wavelength = 3e8 / 32.8e9
    d = wavelength / 2
    n_idx = np.arange(N_elements)
    
    for i in range(num_samples):
        angle_deg = np.random.uniform(-60, 60)
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

    print("--- STEP 3: Normalization (Removing Distance & Ghosting) ---")
    # 1. Log Scale (dB)
    X_dB = 10 * np.log10(X_raw + 1e-15)
    
    # 2. Row-Mean Subtraction (The "Distance Eraser")
    row_means = np.mean(X_dB, axis=1, keepdims=True)
    X_norm = X_dB - row_means

    print("--- STEP 4: Training Random Forest ---")
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_labels, test_size=0.2, random_state=42)
    
    # Random Forest is excellent at handling the "Ghost Peaks" (Non-Linearity)
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, Y_train)
    
    print("--- STEP 5: Evaluation ---")
    preds = model.predict(X_test)
    error = mean_absolute_error(Y_test, preds)
    
    print("="*50)
    print(f"SUCCESS METRIC: Mean Absolute Error = {error:.2f} degrees")
    print("="*50)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:50], 'bo', label='True Angle')
    plt.plot(preds[:50], 'rx', label='AI Prediction')
    plt.title(f"Final System Accuracy (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_solution()
