
from torch import nn,tensor,pi,exp,max,sum

class KpcSoftmax(nn.Module):
    def __init__(self,dim=-1) -> None:
        super().__init__()
        # 默认最后一个维度计算
        self.dim = dim

    def forward(self,x):
        # 整体减去最大值
        """ 
        减去最大值（x - max_x）：在公式 1 中如果不做这个操作，
        当输入 1000.0 时，math.exp(1000) 会直接触发系统浮点数上限返回 inf（无穷大），
        导致整个网络崩掉。减去最大值后，最大项变成 exp(0) = 1，其余项变成负数，exp(负数) 在 (0, 1] 之间，
        完美解决了数值爆炸问题。 
        """
        x_max,_ = max(x,dim=self.dim,keepdim=True)
        x = x - x_max

        # 分子：计算每个元素的指数
        exp_x = exp(x)
        # 计算全部项的和 做分母
        exp_sum = sum(exp_x,dim=self.dim,keepdim=True)
        # 计算比例分布
        softmax_x = exp_x / exp_sum
        return softmax_x

# --- 测试代码 ---
if __name__ == "__main__":
    softmax_layer = KpcSoftmax(dim=-1)
    
    # 模拟一个 Batch 大小为 2，类别数为 3 的输入
    logits = tensor([[1.0, 2.0, 3.0], 
                           [1000.0, 1001.0, 1002.0]], requires_grad=True) # 第二行测试大数稳定性
    
    probs = softmax_layer(logits)
    print("前向传播结果（概率）:\n", probs)
    print("每行概率之和:", probs.sum(dim=-1)) # 期望输出: [1.0, 1.0]
    
    # 反向传播测试
    # loss = probs.sum() # 假设我们要最大化第一个样本属于第三类的概率
    # loss.backward()
    # print("自动求导梯度:\n", logits.grad)