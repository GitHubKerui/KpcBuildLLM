"""Basic tests for layer components."""

import torch
import pytest
from kpc_llm.layers.attention import MultiHeadAttention
from kpc_llm.layers.feed_forward import FeedForward
from kpc_llm.layers.embedding import TokenEmbedding


def test_multi_head_attention():
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    num_heads = 8

    model = MultiHeadAttention(hidden_size, num_heads)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = model(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


def test_feed_forward():
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    intermediate_size = 256

    model = FeedForward(hidden_size, intermediate_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = model(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


def test_token_embedding():
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    hidden_size = 64

    model = TokenEmbedding(vocab_size, hidden_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(token_ids)

    assert output.shape == (batch_size, seq_len, hidden_size)


if __name__ == "__main__":
    pytest.main([__file__])
