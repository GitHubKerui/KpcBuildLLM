"""Feed-forward network implementation."""

import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w2(self.dropout(self.activation(self.w1(x))))
