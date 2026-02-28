import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  THE FIX: RANDOM CODEBOOK GENERATION
#  Why: 1-Bit DFT beams have "ghosts". Random patterns are unique for every angle.
# =============================================================================

def generate_random_codebook(N_elements=64, Codebook_Size=32):
    print(f"Generating Random Codebook: {N_elements} elements, {Codebook_Size} patterns.")
    
    # 1. Randomly choose 0 or 1
    # 2. Map 0 -> 0 radians, 1 -> Pi radians
    random_bits = np.random.randint(0, 2, (N_elements, Codebook_Size))
    codebook_phases = random_bits * np.pi
    
    return codebook_phases

if __name__ == "__main__":
    # Generate
    codebook = generate_random_codebook(N_elements=64, Codebook_Size=32)
    
    # Save (Overwriting the old bad codebook)
    np.save("Phase2_Sensing/codebook.npy", codebook)
    print("Success! New Random Codebook saved.")
    
    # Visualize to confirm it looks like "Static Noise" (which is good!)
    plt.imshow(codebook, aspect='auto', cmap='Greys')
    plt.title("Random Codebook (The 'Fingerprint' Generator)")
    plt.xlabel("Beam Index")
    plt.ylabel("IRS Element Index")
    plt.show()