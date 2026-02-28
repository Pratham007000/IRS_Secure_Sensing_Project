import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
#  PHASE 2 COMPLETE PIPELINE: GENERATION + NORMALIZATION + TRAINING
# =============================================================================

def run_pipeline():
    print("--- STEP 1: Generating Random Codebook ---")
    N_elements = 64
    Num_Beams = 32
    # Create random phases (0 or Pi)
    codebook_phases = np.random.randint(0, 2, (N_elements, Num_Beams)) * np.pi
    
    print("--- STEP 2: Simulating 5,000 Users ---")
    num_samples = 5000
    X_data = np.zeros((num_samples, Num_Beams))
    Y_labels = np.zeros((num_samples,)) # 1D array for scikit-learn
    
    # Pre-calculate constants for speed
    wavelength = 3e8 / 32.8e9
    d = wavelength / 2
    n_idx = np.arange(N_elements)
    
    # Hardware Amplitude Model (The Dip)
    # If phase is Pi (ON) -> Amp 0.4. If 0 (OFF) -> Amp 0.9
    w_amps = np.where(codebook_phases > 1.0, 0.4, 0.9)
    W_complex = w_amps * np.exp(1j * codebook_phases)

    for i in range(num_samples):
        # Random User
        angle_deg = np.random.uniform(-60, 60)
        dist = np.random.uniform(2, 10)
        Y_labels[i] = angle_deg
        
        # Channel Generation (ULA Model)
        angle_rad = np.deg2rad(angle_deg)
        spatial_freq = (2 * np.pi * d / wavelength) * np.sin(angle_rad)
        h = np.exp(1j * n_idx * spatial_freq) * (1/dist) # Simple path loss
        
        # Sensing (The Scan)
        # y = |h * W|^2
        # Result is vector of 32 signal strengths
        signal = np.abs(h @ W_complex)**2
        
        # Add Noise
        noise = np.random.normal(0, 1e-6, Num_Beams)
        X_data[i, :] = signal + noise

    print("--- STEP 3: The Critical Fix (Row-Wise Normalization) ---")
    # We divide every sample by its own max value.
    # This removes the effect of 'Distance'. All signals range 0.0 to 1.0.
    # The AI now only sees the SHAPE, not the volume.
    row_maxes = X_data.max(axis=1, keepdims=True)
    X_norm = X_data / row_maxes
    
    print(f"Data Shape: {X_norm.shape}. Range: {X_norm.min()} to {X_norm.max()}")

    print("--- STEP 4: Training the AI ---")
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_labels, test_size=0.2, random_state=42)
    
    # Robust Architecture
    model = MLPRegressor(hidden_layer_sizes=(512, 256), # Simplified but wide
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         max_iter=1000,
                         random_state=42)
    
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
    plt.plot(preds[:50], 'rx', label='AI Prediction')
    plt.title(f"Sensing Accuracy (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_pipeline()
