"""Embedding layer implementation."""

import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, token_ids):
        return self.embedding(token_ids) * (self.hidden_size ** 0.5)
