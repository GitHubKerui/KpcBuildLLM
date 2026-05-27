from torch import nn,Tensor,manual_seed

class KpcTransformerBlock(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()

    def forward(self,x):
        out = x
        return out


