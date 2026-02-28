import numpy as np
import matplotlib.pyplot as plt

class IndustrialIRSChannel:
    def __init__(self, N_elements, frequency=32.8e9):
        self.N = N_elements
        self.fc = frequency
        self.wavelength = 3e8 / frequency
        self.d = self.wavelength / 2

    def generate_channel_vector(self, distance, is_los=True, k_factor_db=10):
        # Path Loss
        path_loss_linear = (self.wavelength / (4 * np.pi * distance)) ** 2
        
        # Rician Fading
        if is_los:
            k_linear = 10 ** (k_factor_db / 10)
            h_los = np.ones(self.N, dtype=complex)
            h_nlos = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)
            h_fading = np.sqrt(k_linear / (k_linear + 1)) * h_los + \
                       np.sqrt(1 / (k_linear + 1)) * h_nlos
        else:
            h_fading = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)

        return np.sqrt(path_loss_linear) * h_fading

    def get_1bit_reflection_matrix(self, ideal_phases):
        # 1. Normalize phases
        phases_norm = np.angle(np.exp(1j * ideal_phases)) % (2 * np.pi)
        
        # 2. Quantize: 0 or Pi
        discrete_phases = np.where(
            (phases_norm > np.pi/2) & (phases_norm < 3*np.pi/2), 
            np.pi, 
            0.0
        )

        # 3. Amplitude Dip (Resonance Loss)
        # OFF (0 rad) -> 0.9 reflection
        # ON (Pi rad) -> 0.4 reflection (The "Dip")
        amplitudes = np.where(discrete_phases > 1.0, 0.4, 0.9)

        reflection_coeffs = amplitudes * np.exp(1j * discrete_phases)
        return np.diag(reflection_coeffs), reflection_coeffs

if __name__ == "__main__":
    irs_sim = IndustrialIRSChannel(N_elements=50)
    ideal_phases = np.linspace(0, 2*np.pi, 50)
    _, real_coeffs = irs_sim.get_1bit_reflection_matrix(ideal_phases)
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Phase
    plt.subplot(1, 2, 1)
    plt.plot(ideal_phases, np.degrees(ideal_phases), label='Ideal Request', linestyle='--')
    plt.step(ideal_phases, np.degrees(np.angle(real_coeffs)), label='1-Bit Hardware', where='mid', linewidth=2)
    plt.title("Constraint 1: Phase Quantization")
    plt.xlabel("Requested Phase (rad)")
    plt.ylabel("Actual Phase (deg)")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Amplitude Dip
    plt.subplot(1, 2, 2)
    plt.plot(ideal_phases, np.abs(real_coeffs), color='red', marker='o')
    plt.title("Constraint 2: Amplitude Dip (Loss)")
    plt.xlabel("Requested Phase")
    plt.ylabel("Reflection Amplitude")
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    print("Simulation Complete. A window with two graphs should appear.")
    plt.tight_layout()
    plt.show()
