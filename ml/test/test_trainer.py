import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.model import TranADNetwork
from ml.torch_dataset import WindowDataset
from ml.trainer import Trainer


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Fake training data
windows = np.random.randn(
    100,
    120,
    22,
).astype(np.float32)


dataset = WindowDataset(windows)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)


model = TranADNetwork(
    input_size=22,
    hidden_size=64,
    num_heads=4,
)


trainer = Trainer(
    model=model,
    learning_rate=1e-3,
)


print("Device:")
print(trainer.device)


print("\nTraining:")

for epoch in range(5):

    loss = trainer.train_epoch(
        dataloader
    )

    print(
        f"Epoch {epoch + 1}: "
        f"loss={loss:.6f}"
    )