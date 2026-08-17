import numpy as np
import torch

from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


MODEL_PATH = "artifacts/attention_reconstruction_v1.pt"


# -----------------------------
# Load model
# -----------------------------

model = TranADNetwork(
    input_size=22,
    hidden_size=64,
    num_heads=4,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)


# -----------------------------
# Create scorer
# -----------------------------

scorer = AnomalyScorer(
    model,
    device="cpu",
)


# -----------------------------
# Fake test windows
# -----------------------------

normal_windows = np.random.normal(
    0,
    0.1,
    size=(10, 120, 22),
).astype(np.float32)


scores = scorer.score(
    normal_windows
)


print("Scores:")
print(scores)

print("\nShape:")
print(scores.shape)

print("\nMinimum:")
print(scores.min())

print("\nMaximum:")
print(scores.max())

print("\nMean:")
print(scores.mean())