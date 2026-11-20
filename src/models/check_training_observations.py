import numpy as np

# Load existing tensors
path = "Data/processed/teacher_tensors/"

x = np.load(path + "model_input_tensor.npy")
y = np.load(path + "ground_truth_tensor.npy")
mask = np.load(path + "observation_mask_tensor.npy")

# Chronological training split
train_end = 18412

y_train = y[:train_end]
mask_train = mask[:train_end]

# Valid observed values only
valid_train = (
    (mask_train == 1) &
    np.isfinite(y_train)
)

print("Training tensor:", y_train.shape)
print("Valid observed values:", valid_train.sum())

print("\nFeature-wise valid observation counts:")

feature_names = [
    "PM2.5", "PM10", "NO2", "O3", "SO2", "NOx", "CO",
    "Temperature", "Relative Humidity", "Wind Speed",
    "Wind Direction", "Pressure", "Rainfall"
]

for i, name in enumerate(feature_names):
    count = valid_train[:, :, i].sum()
    print(f"{name:22s}: {count}")