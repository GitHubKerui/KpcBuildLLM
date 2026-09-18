
from torch import nn,Tensor,manual_seed,ones,zeros,sqrt

class KpcNormal(nn.Module):
    def __init__(self,emb_dim,unbiased=False) -> None:
        super().__init__()
        self.unbiased = unbiased
        self.eps = 1e-5
        # 个人觉得这个weight 和 bias 没必要加 因为在最后一层有linear层做特征捕捉。这个没必要
        self.weight = nn.Parameter(ones(emb_dim))
        self.bias = nn.Parameter(zeros(emb_dim))

    def forward(self,x):
        # 1 算出最后一个维度的均值 
        mean = x.mean(dim=-1,keepdim=True)
        # 2 算出方差最后一个维度的方差，用了有偏估计的设置。
        var = x.var(dim=-1,keepdim=True,unbiased = self.unbiased)
        varsqrt = sqrt(var + self.eps)
        # 3 整个网络拉回中心0点，均方差 1.
        x_norm = (x - mean) / varsqrt
        # 这里在最后一层有个 linear层 我觉得没必要加 这个
        out = self.weight * x_norm + self.bias
        return out
