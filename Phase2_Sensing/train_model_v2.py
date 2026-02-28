import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# =============================================================================
#  PHASE 2, WEEK 5: ADVANCED AI TRAINING (V2)
#  Goal: Reduce error from ~30 deg to < 5 deg using Deep Learning.
# =============================================================================

def train_advanced_ai():
    print("1. Loading Dataset...")
    try:
        data = np.load("Phase2_Sensing/sensing_dataset.npz")
        X = data['X']
        Y = data['Y'].ravel()
    except FileNotFoundError:
        print("Error: Dataset missing.")
        return

    # 2. Advanced Preprocessing (Standardization)
    # This forces the data to have Mean=0 and Variance=1.
    # It helps the Neural Network see "subtle" differences in the 1-bit noise.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    print(f"Data Ready. Training on {len(X_train)} samples.")
    
    # 3. Deep Neural Network Architecture (The "Big Brain")
    # - More Layers: (512, 256, 128) -> Deep Learning
    # - Solver: 'adam' (Adaptive Moment Estimation)
    # - Alpha: 0.0001 (Regularization to prevent memorizing)
    model = MLPRegressor(hidden_layer_sizes=(512, 256, 128),
                         activation='relu',
                         solver='adam',
                         alpha=0.0001,
                         learning_rate_init=0.001,
                         max_iter=2000,  # Train longer
                         early_stopping=True, # Stop if it stops learning
                         random_state=42)
    
    # 4. Train
    print("Training Deep Neural Network... (This might take 10-20 seconds)")
    model.fit(X_train, Y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    error = mean_absolute_error(Y_test, predictions)
    
    print("="*40)
    print(f"UPGRADED RESULTS: Mean Absolute Error = {error:.2f} degrees")
    print("="*40)
    
    # 6. Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:50], 'bo-', label='True Location')
    plt.plot(predictions[:50], 'rx--', label='AI Prediction')
    plt.title(f"Deep Learning Performance (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_advanced_ai()