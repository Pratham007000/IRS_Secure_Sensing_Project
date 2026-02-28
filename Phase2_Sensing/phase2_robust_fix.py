import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
#  PHASE 2 ROBUST FIX: RANDOM FOREST + DISTANCE REMOVAL
# =============================================================================

def run_robust_pipeline():
    print("--- STEP 1: Generating Codebook (Random) ---")
    N_elements = 64
    Num_Beams = 64
    codebook_phases = np.random.randint(0, 2, (N_elements, Num_Beams)) * np.pi
    
    # Hardware Model
    w_amps = np.where(codebook_phases > 1.0, 0.4, 0.9)
    W_complex = w_amps * np.exp(1j * codebook_phases)

    print("--- STEP 2: Simulating 5,000 Users ---")
    num_samples = 5000
    X_raw = np.zeros((num_samples, Num_Beams))
    Y_labels = np.zeros((num_samples,))
    
    # Physics
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

    print("--- STEP 3: The 'Distance Removal' Fix ---")
    # 1. Convert to Decibels
    X_dB = 10 * np.log10(X_raw + 1e-15)
    
    # 2. Subtract the Mean of each ROW
    # This mathematically cancels out the path loss (1/dist^2).
    # Now, a user at 2m and 10m have the EXACT same values.
    row_means = np.mean(X_dB, axis=1, keepdims=True)
    X_norm = X_dB - row_means
    
    print(f"Data Normalized. Shape: {X_norm.shape}")

    print("--- STEP 4: Training Random Forest (The Workhorse) ---")
    # Random Forest is often better than Neural Networks for this specific data
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_labels, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, Y_train)
    
    print("--- STEP 5: Evaluation ---")
    preds = model.predict(X_test)
    error = mean_absolute_error(Y_test, preds)
    
    print("="*50)
    print(f"ROBUST ACCURACY: Mean Absolute Error = {error:.2f} degrees")
    print("="*50)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:50], 'bo', label='True Angle')
    plt.plot(preds[:50], 'rx', label='RF Prediction')
    plt.title(f"Random Forest Performance (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_robust_pipeline()
