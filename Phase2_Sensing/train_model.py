import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
#  PHASE 2, WEEK 5: TRAIN THE SENSING AI
#  Goal: Train a Neural Network to predict user angle from IRS "Fingerprints".
# =============================================================================

def train_sensing_ai():
    print("1. Loading Dataset...")
    try:
        data = np.load("Phase2_Sensing/sensing_dataset.npz")
        X = data['X']  # The Sensing Vectors (Input)
        Y = data['Y'].ravel()  # The Angles (Target Output)
    except FileNotFoundError:
        print("Error: Dataset not found! Run generate_sensing_data.py first.")
        return

    # 2. Preprocessing
    # Normalize inputs to help the Neural Network learn faster
    # We divide by the max signal strength to keep values between 0 and 1
    X_max = np.max(X)
    X_normalized = X / X_max
    
    # Split: 80% for Training, 20% for Testing (to see if it truly learned)
    X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=42)
    
    print(f"Data Loaded: {len(X)} samples.")
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # 3. Define the AI Model (Deep Neural Network)
    # Architecture: Input(32) -> Hidden(128) -> Hidden(64) -> Output(1)
    # 'relu': Standard activation function for deep learning
    # 'adam': The optimizer that adjusts weights
    model = MLPRegressor(hidden_layer_sizes=(128, 64),
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         max_iter=500,
                         random_state=42)
    
    # 4. Train
    print("Training Neural Network... (This may take a few seconds)")
    model.fit(X_train, Y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    error = mean_absolute_error(Y_test, predictions)
    
    print("="*40)
    print(f"RESULTS: Mean Absolute Error = {error:.2f} degrees")
    print("="*40)
    
    # 6. Visualize Success
    plt.figure(figsize=(10, 5))
    # Plot first 50 test cases for clarity
    plt.plot(Y_test[:50], 'bo-', label='True Location')
    plt.plot(predictions[:50], 'rx--', label='AI Prediction')
    plt.title(f"AI Performance (Error: {error:.2f} deg)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_sensing_ai()