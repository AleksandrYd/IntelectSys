import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_layers: int, hidden: int, dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_features = 64
        for _ in range(n_layers):
            blocks.append(nn.Linear(in_features, hidden))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            in_features = hidden
        blocks.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)