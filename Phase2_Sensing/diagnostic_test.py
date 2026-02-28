import numpy as np
import matplotlib.pyplot as plt

def test_physics():
    print("Running Physics Diagnostic...")
    
    # Setup
    N_elements = 64
    wavelength = 3e8 / 32.8e9
    d = wavelength / 2
    n_idx = np.arange(N_elements)
    
    # Create a Test Beam (Beam #8)
    # This mathematically steers the beam to ~14 degrees
    k = 8
    ideal_phases = 2 * np.pi * k * n_idx / N_elements
    
    # Apply 1-Bit Quantization (0 or Pi)
    phases_norm = ideal_phases % (2 * np.pi)
    w_phases = np.where((phases_norm > np.pi/2) & (phases_norm < 3*np.pi/2), np.pi, 0.0)
    
    # Apply Amplitude Dip (The "Hardware Constraint")
    w_amps = np.where(w_phases > 1.0, 0.4, 0.9)
    W_complex = w_amps * np.exp(1j * w_phases)
    
    # Sweep User Angle (-60 to +60)
    angles = np.linspace(-60, 60, 300)
    responses = []
    
    for angle_deg in angles:
        # Channel Model
        angle_rad = np.deg2rad(angle_deg)
        spatial_freq = (2 * np.pi * d / wavelength) * np.sin(angle_rad)
        h = np.exp(1j * n_idx * spatial_freq)
        
        # Sensing
        signal = np.abs(np.dot(h, W_complex))**2
        responses.append(signal)
        
    # Analyze
    peak_angle = angles[np.argmax(responses)]
    max_sig = np.max(responses)
    min_sig = np.min(responses)
    contrast = max_sig - min_sig
    
    print(f"Peak Signal found at: {peak_angle:.2f} degrees")
    print(f"Signal Contrast: {contrast:.2f}")

    if contrast < 10.0:
        print("FAIL: The response is too flat. The IRS is essentially a mirror.")
    else:
        print("PASS: Strong beam detected! The Physics Engine is working.")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(angles, responses, linewidth=2)
    plt.title(f"Diagnostic: Beam {k} Pattern")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Signal Strength")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_physics()
