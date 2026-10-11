
from torch import nn

class KpcFeedForward(nn.Module):
    def __init__(self,embed_dim=512) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.layer = nn.Sequential(
            nn.Linear(embed_dim,embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4,embed_dim)
        )

    def forward(self,x):
        return self.layer(x)