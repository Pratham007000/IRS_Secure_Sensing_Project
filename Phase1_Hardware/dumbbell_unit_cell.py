import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  PHASE 1, WEEK 3: DUMBBELL UNIT CELL FREQUENCY RESPONSE
#  Reference: Paper W1 (Fig 11) - Resonance at 32.8 GHz
# =============================================================================

def unit_cell_resonance_model(freq_range_hz, center_freq=32.8e9, q_factor=15):
    """
    Models the reflection amplitude of the IRS unit cell across frequencies.
    The 'Dip' happens because of resonance.
    
    Model: Inverted Lorentzian function (standard for resonance absorption).
    At center_freq, reflection is lowest (0.4). Far away, it returns to 0.9.
    """
    # Normalized frequency deviation
    delta = (freq_range_hz - center_freq) / center_freq
    
    # Lorentzian Shape: 1 / (1 + x^2)
    # We invert it to make a dip.
    resonance_curve = 1 / (1 + (q_factor * delta)**2)
    
    # Max Reflection (Off Resonance) = 0.9
    # Min Reflection (At Resonance) = 0.4
    # Depth of dip = 0.9 - 0.4 = 0.5
    
    reflection_amplitude = 0.9 - (0.5 * resonance_curve)
    
    return reflection_amplitude

# =============================================================================
#  RUN SIMULATION
# =============================================================================
if __name__ == "__main__":
    # 1. Define Frequency Range (28 GHz to 38 GHz)
    freqs = np.linspace(28e9, 38e9, 500)
    
    # 2. Compute Response
    amplitude_response = unit_cell_resonance_model(freqs)
    
    # 3. Plotting
    plt.figure(figsize=(9, 6))
    plt.plot(freqs / 1e9, amplitude_response, linewidth=3, color='purple')
    
    # Highlight the Operating Point (32.8 GHz)
    plt.axvline(x=32.8, color='red', linestyle='--', label='Resonance Freq (32.8 GHz)')
    plt.scatter([32.8], [0.4], color='red', s=100, zorder=5)
    plt.text(33, 0.45, "Max Loss (Amp = 0.4)\nPhase is 180° here", color='red', fontweight='bold')
    
    plt.title("Week 3 Deliverable: Frequency Selective Reflection")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflection Magnitude |Gamma|")
    plt.ylim(0, 1.0)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    print("Generating Frequency Response Plot...")
    plt.tight_layout()
    plt.show()