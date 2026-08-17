import numpy as np

from ml.baseline import MeanBaseline


rng = np.random.default_rng(42)

windows = rng.normal(
    size=(100, 120, 22)
).astype(np.float32)

model = MeanBaseline()

model.fit(windows)

scores = model.score(windows)

print("Training windows:", windows.shape)
print("Scores shape:", scores.shape)
print("First 10 scores:", scores[:10])
print("Minimum score:", scores.min())
print("Maximum score:", scores.max())