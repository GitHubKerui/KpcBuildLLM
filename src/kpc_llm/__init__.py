"""KpcLLM - A personal LLM architecture implementation."""

__version__ = "0.1.0"

from kpc_llm.core.transformer import TransformerBlock
from kpc_llm.layers.attention import MultiHeadAttention
from kpc_llm.layers.feed_forward import FeedForward
from kpc_llm.layers.embedding import TokenEmbedding

__all__ = [
    "TransformerBlock",
    "MultiHeadAttention",
    "FeedForward",
    "TokenEmbedding",
]
