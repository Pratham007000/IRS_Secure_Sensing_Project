import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  PHASE 2, WEEK 4: IRS CODEBOOK DESIGN (DFT-BASED)
#  Goal: Create 32 distinct beam patterns to scan the environment.
# =============================================================================

def generate_dft_codebook(N_elements=256, Codebook_Size=32):
    """
    Generates a Codebook based on the Discrete Fourier Transform (DFT).
    DFT beams are naturally orthogonal (spaced out perfectly).
    
    Args:
        N_elements: Number of unit cells on the IRS (e.g., 16x16 = 256).
        Codebook_Size: Number of distinct beams we want (e.g., 32).
    """
    print(f"Generating DFT Codebook for {N_elements} elements with {Codebook_Size} patterns...")
    
    # 1. Create standard DFT Matrix (size N x N)
    # The 'k-th' column represents a beam pointing at a specific angle.
    dft_matrix = np.fft.fft(np.eye(N_elements))
    
    # 2. Select specific columns to form our codebook
    # We pick 'Codebook_Size' columns evenly spaced to cover the field of view.
    indices = np.linspace(0, N_elements, Codebook_Size, endpoint=False, dtype=int)
    ideal_codebook = dft_matrix[:, indices]
    
    # 3. APPLY HARDWARE CONSTRAINTS (The "1-Bit" Logic)
    # We must quantize the DFT phases to 0 or Pi.
    # Logic: If angle is in [90, 270], it becomes Pi. Else 0.
    
    phases = np.angle(ideal_codebook) % (2 * np.pi)
    
    # Quantize to 0 (OFF) or Pi (ON)
    quantized_phases = np.where(
        (phases > np.pi/2) & (phases < 3*np.pi/2), 
        np.pi, 
        0.0
    )
    
    return quantized_phases

def plot_beam_directions(codebook):
    """
    Visualizes the first few patterns to show they are different.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the phase settings for the first 5 beams
    for i in range(5):
        plt.plot(codebook[:, i] + (i * 4), label=f'Beam {i+1}')
        
    plt.title("Visualizing First 5 Beam Patterns (Offset for Clarity)")
    plt.xlabel("IRS Element Index")
    plt.ylabel("Phase State (0 or Pi) + Offset")
    plt.yticks([]) # Hide y-axis numbers
    plt.legend(loc='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Settings
    N = 64  # Using 64 elements for the prototype size
    K = 32  # 32 distinct beams
    
    # Generate
    codebook = generate_dft_codebook(N, K)
    
    # Save to file (This is the "Menu" our IRS will use later)
    np.save("Phase2_Sensing/codebook.npy", codebook)
    print(f"Success! Codebook saved to 'Phase2_Sensing/codebook.npy'. Shape: {codebook.shape}")
    
    # Visualize
    plot_beam_directions(codebook)