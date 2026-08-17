import numpy as np

from ml.torch_dataset import WindowDataset


windows = np.random.randn(
    100,
    120,
    22,
).astype(np.float32)


dataset = WindowDataset(windows)

print("Dataset length:")
print(len(dataset))

print("\nFirst sample shape:")
print(dataset[0].shape)

print("\nFirst sample dtype:")
print(dataset[0].dtype)