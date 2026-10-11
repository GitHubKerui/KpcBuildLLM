from torch import nn,rand,softmax,manual_seed,Tensor
from kpc_llm.utils.logger import getlogger



logger = getlogger()
class SelfAttentionV1(nn.Module):
    def __init__(self,in_d,out_d) -> None:
        super().__init__()
        self.W_q = nn.Parameter(rand(in_d,out_d))
        self.W_k = nn.Parameter(rand(in_d,out_d))
        self.W_v = nn.Parameter(rand(in_d,out_d))
        

    def forward(self,input):
        #把token的嵌入向量投影到（转化到）三个 query,key,value特征空间 可以设置特征空间的shape，一般是升维的 step 1
        in_queries = input @ self.W_q
        in_keys = input @ self.W_k
        in_values = input @ self.W_v

        #token输入的query空间特征，和key空间特征计算出注意力分数 step 2
        attention_scores = in_queries @ in_keys.T
        
        #对注意力分数样本空间进行缩放scale,这里的注意力分数因为n*n 所以方差是 n ，标准差是 n^0.5 ,n的开方所以
        attention_weight = softmax(attention_scores / in_queries.shape[-1]**0.5 , dim=-1)
        context_vector = attention_weight @ in_values

        return context_vector

if __name__ == '__main__':
    manual_seed(517)
    inputs = Tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )
    self_attention_v1 = SelfAttentionV1(3,6)
    logger.info(f'self_attention_v1 : {self_attention_v1(inputs)}')
    