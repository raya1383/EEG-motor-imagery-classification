import numpy as np

data = np.load("data/processed/ds1a_extracted.npz")

print("Keys inside file:")
print(data.files)

print("\nX shape:", data["X"].shape)
print("y shape:", data["y"].shape)

print("\nTrain shape:", data["X_train"].shape)
print("Test shape:", data["X_test"].shape)
