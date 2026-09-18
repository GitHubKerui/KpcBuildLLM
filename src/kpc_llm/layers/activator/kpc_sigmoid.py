
from torch import nn,tensor,pi,exp

class KpcSigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        out = 1.0 /(1.0 + exp(-x))
        return out

# --- 测试代码 ---
if __name__ == "__main__":
    advanced_sigmoid = KpcSigmoid()
    x = tensor([-2.0, 0.0, 2.0], requires_grad=True)
    
    y = advanced_sigmoid(x)
    print("手动算子前向结果:", y)
    
    loss = y.sum()
    loss.backward()
    print("手动实现的反向梯度:", x.grad)