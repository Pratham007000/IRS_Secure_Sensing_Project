import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def run_db_pipeline():
    print("--- STEP 1: Generating High-Res Random Codebook (64 Beams) ---")
    N_elements = 64
    Num_Beams = 64  # High resolution
    
    # Random Phases (0 or Pi)
    codebook_phases = np.random.randint(0, 2, (N_elements, Num_Beams)) * np.pi
    
    print("--- STEP 2: Simulating 5,000 Users ---")
    num_samples = 5000
    X_raw = np.zeros((num_samples, Num_Beams))
    Y_labels = np.zeros((num_samples,))
    
    # Physics Constants
    wavelength = 3e8 / 32.8e9
    d = wavelength / 2
    n_idx = np.arange(N_elements)
    
    # Hardware Model (Amplitude Dip)
    w_amps = np.where(codebook_phases > 1.0, 0.4, 0.9)
    W_complex = w_amps * np.exp(1j * codebook_phases)

    for i in range(num_samples):
        # Random User (-60 to +60 deg)
        angle_deg = np.random.uniform(-60, 60)
        dist = np.random.uniform(2, 10)
        Y_labels[i] = angle_deg
        
        # Channel
        angle_rad = np.deg2rad(angle_deg)
        spatial_freq = (2 * np.pi * d / wavelength) * np.sin(angle_rad)
        h = np.exp(1j * n_idx * spatial_freq) * (1/dist)
        
        # Sensing
        signal_linear = np.abs(h @ W_complex)**2
        
        # Add Noise and Convert to Decibels (The Fix)
        # We add 1e-15 to avoid log(0)
        noise = np.random.normal(0, 1e-10, Num_Beams)
        X_raw[i, :] = 10 * np.log10(signal_linear + np.abs(noise) + 1e-15)

    print(f"--- STEP 3: Data Ready (Shape: {X_raw.shape}) ---")
    
    # Standardize for AI
    scaler = StandardScaler()
    X_ready = scaler.fit_transform(X_raw)

    print("--- STEP 4: Training the AI ---")
    X_train, X_test, Y_train, Y_test = train_test_split(X_ready, Y_labels, test_size=0.2, random_state=42)
    
    model = MLPRegressor(hidden_layer_sizes=(512, 256, 128),
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         max_iter=1000,
                         early_stopping=True,
                         random_state=42)
    
    model.fit(X_train, Y_train)
    
    print("--- STEP 5: Evaluation ---")
    preds = model.predict(X_test)
    error = mean_absolute_error(Y_test, preds)
    
    print("="*50)
    print(f"FINAL ACCURACY: Mean Absolute Error = {error:.2f} degrees")
    print("="*50)
    
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:50], 'bo', label='True Angle')
    plt.plot(preds[:50], 'r.', label='AI Prediction')
    plt.title(f"Final Performance (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_db_pipeline()
