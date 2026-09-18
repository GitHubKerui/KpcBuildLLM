from torch import nn,tensor,tanh,pi,sqrt,pow

""" 
In practice, it's common to implement a computationally cheaper approximation: 
$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right]\right)
$ (the original GPT-2 model was also trained with this approximation) 
"""

""" 
其实这里用 nn.GELU()就可以直接添加到 nn.Sequential里。
"""
class KpcGELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        out = 0.5 * x * (1 + tanh(sqrt(tensor(2.0/pi)) * (x + 0.044715 * pow(x,3))))
        return out

# --- 测试代码 ---
if __name__ == "__main__":
    kcpGl = KpcGELU()
    x = tensor([-0.5,-30,1,2,3,30,300])
    out = kcpGl(x)
    print(f"输入 ： {x}")
    print(f"结果 ： {out}")