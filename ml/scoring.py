import torch
import numpy as np


class AnomalyScorer:

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def score(self, windows, batch_size=2048):

        scores_list = []
        num_windows = len(windows)

        for i in range(0, num_windows, batch_size):
            batch = windows[i : i + batch_size]

            if isinstance(batch, np.ndarray):
                batch = torch.tensor(
                    batch,
                    dtype=torch.float32,
                )

            batch = batch.to(self.device)

            reconstructed = self.model(batch)

            # Mean squared reconstruction error
            errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=(1, 2),
            )

            scores_list.append(errors.cpu().numpy())

        return np.concatenate(scores_list, axis=0)