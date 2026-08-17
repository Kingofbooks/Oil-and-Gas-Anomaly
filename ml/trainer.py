import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:

    def __init__(self,model: nn.Module,learning_rate: float = 1e-3,device: str | None = None):
        self.model = model
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.model.to(self.device)
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)

    def train_epoch(self,dataloader: DataLoader) -> float:

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for windows in dataloader:
            windows = windows.to(self.device)
            reconstruction = self.model(windows)
            loss = self.criterion(reconstruction,windows)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_size = windows.size(0)
            total_loss += (loss.item() * batch_size)
            total_samples += batch_size

        return total_loss / total_samples