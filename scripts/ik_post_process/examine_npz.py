import numpy as np
import pickle

# Load the npz file and examine its structure
data = np.load('data/demo_npzs/taijiquan_female_ID0_difTraj_raw.npz')

print("Keys in the npz file:")
for key in data.files:
    print(f"  {key}: {data[key].shape} {data[key].dtype}")

print("\nDetailed information:")
for key in data.files:
    print(f"\n{key}:")
    print(f"  Shape: {data[key].shape}")
    print(f"  Data type: {data[key].dtype}")
    print(f"  Min value: {np.min(data[key])}")
    print(f"  Max value: {np.max(data[key])}")
    print(f"  Mean value: {np.mean(data[key])}")