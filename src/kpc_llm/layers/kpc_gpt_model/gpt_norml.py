
from torch import nn,Tensor,manual_seed

class KpcFinalNormal(nn.Module):
    def __init__(self,emb_dim) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,x):
        out = x
        return out
