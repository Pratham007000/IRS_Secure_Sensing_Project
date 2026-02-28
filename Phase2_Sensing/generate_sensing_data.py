import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  PHASE 2: SENSING DATA GENERATION
#  Goal: Simulate 5,000 users and record what the IRS "sees" (Sensing Vector).
# =============================================================================

def generate_channel(N_elements, angle_deg, distance):
    """
    Generates the channel vector h for a user at a specific angle and distance.
    Uses a simplified Far-Field model for speed.
    """
    wavelength = 3e8 / 32.8e9  # 32.8 GHz
    d = wavelength / 2
    
    # spatial frequency
    angle_rad = np.deg2rad(angle_deg)
    spatial_freq = (2 * np.pi * d / wavelength) * np.sin(angle_rad)
    
    # Array Response Vector (Steering Vector)
    n = np.arange(N_elements)
    h_los = np.exp(1j * n * spatial_freq)
    
    # Add Path Loss (Signal gets weaker with distance)
    path_loss = 1 / (distance ** 2)
    
    return h_los * np.sqrt(path_loss)

def generate_dataset(num_samples=5000):
    # 1. Load the Codebook we just made
    try:
        codebook = np.load("Phase2_Sensing/codebook.npy")
        print(f"Loaded Codebook: {codebook.shape}")
    except FileNotFoundError:
        print("Error: codebook.npy not found! Run generate_codebook.py first.")
        return

    N_elements, Num_Beams = codebook.shape
    
    # Arrays to store data
    # X = The "Sensing Vector" (32 signal strengths)
    # Y = The "Label" (The actual Angle of the user)
    X_data = np.zeros((num_samples, Num_Beams))
    Y_labels = np.zeros((num_samples, 1))
    
    print(f"Generating {num_samples} samples...")
    
    for i in range(num_samples):
        # 2. Random User Location
        # Angle: -60 to +60 degrees
        true_angle = np.random.uniform(-60, 60)
        # Distance: 2 to 10 meters
        true_dist = np.random.uniform(2, 10)
        
        Y_labels[i] = true_angle
        
        # 3. Generate Channel (Physics)
        h = generate_channel(N_elements, true_angle, true_dist)
        
        # 4. The "Scan" (Math: y = |h^T * W|^2)
        # We test all 32 beams against this one user channel
        # The IRS hardware phases are in the codebook (0 or Pi)
        # We convert phases to complex numbers: exp(j * phase)
        # But wait! Remember the "Amplitude Dip"?
        # If phase is Pi (ON), amp is 0.4. If 0 (OFF), amp is 0.9.
        
        w_phases = codebook  # Matrix of 0s and Pis
        
        # Apply Amplitude Dip Model (Phase 1 Logic!)
        w_amps = np.where(w_phases > 1.0, 0.4, 0.9)
        
        # Combine Amp and Phase into Complex Reflection Coefficients
        W_complex = w_amps * np.exp(1j * w_phases)
        
        # Compute Received Signal for all 32 beams
        # Result is a vector of 32 numbers
        sensing_vector = np.abs(h @ W_complex)**2
        
        # Add some noise (Real world is noisy)
        noise = np.random.normal(0, 0.00001, Num_Beams)
        X_data[i, :] = sensing_vector + noise

    # 5. Save Dataset
    np.savez("Phase2_Sensing/sensing_dataset.npz", X=X_data, Y=Y_labels)
    print("Success! Dataset saved to 'Phase2_Sensing/sensing_dataset.npz'")
    
    return X_data, Y_labels

if __name__ == "__main__":
    X, Y = generate_dataset()
    
    # Visualization: Show one random sample
    plt.figure(figsize=(10, 5))
    plt.bar(range(32), X[0])
    plt.title(f"Example Sensing Vector (User at {Y[0][0]:.2f} deg)")
    plt.xlabel("Beam Index (1-32)")
    plt.ylabel("Received Signal Strength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()