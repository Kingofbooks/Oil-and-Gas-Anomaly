import torch
import torch.nn as nn


class TranADNetwork(nn.Module):

    def __init__(self,input_size: int = 22,hidden_size: int = 64,num_heads: int = 4):
        super().__init__()
        self.input_projection = nn.Linear(input_size,hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,num_heads=num_heads,batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size,input_size)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        attended, _ = self.attention(h,h,h)
        h = self.norm(h + attended)
        reconstruction = (self.output_projection(h))

        return reconstruction