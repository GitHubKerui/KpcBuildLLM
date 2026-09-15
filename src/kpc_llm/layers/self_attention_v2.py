
from torch.nn import Module,Linear
from torch import manual_seed,softmax,Tensor
from kpc_llm.utils.logger import getlogger

logger = getlogger()

class SelfAttentionV2(Module):
    def __init__(self,in_d,out_d) -> None:
        super().__init__()
        self.W_q = Linear(in_d,out_d,bias=False)
        self.W_k = Linear(in_d,out_d,bias=False)
        self.W_v = Linear(in_d,out_d,bias=False)

    def forward(self,input):
        # queries = self.W_q(input)
        # keys = self.W_k(input)
        # values = self.W_v(input)
        # attention_scores = queries @ keys.T
        # attention_weights = softmax(attention_scores / queries.shape[-1]**0.5 , dim = -1)
        # context_vector = attention_weights @ values
        context_vector = self.getContextVectors(input)
        return context_vector

    
    def getAttentionScores(self,input):
        '''
        step 1 输入空间的样本的高维投影化，三个高纬投影空间 query 空间，key空间，value空间correspondding了流程中的三个顺序操作。
        '''
        queries = self.W_q(input)
        keys = self.W_k(input)
        attention_scores = queries @ keys.T
        return attention_scores

    def getAttentionWeight(self,input):
        '''
        step 2 scale 归一化，normalization，正态化，缩放，分布空间缩放，dot计算后的样本方差量值是维度d，标差是 d**0.5，
        因为每个token样本的方差是1 ，那么N个样本的点乘的样本的方差 是 1^2 *N = 1 *N = N 等于token的个数，也就是自注意力方阵的dim。
        所以做归一化 必须除以标准差，等于除，dim^0.5,当然这之前
        '''
        attention_scores = self.getAttentionScores(input)
        dim_num = attention_scores.shape[-1]
        attention_weights = softmax(attention_scores/dim_num**0.5 , dim = -1)
        return attention_weights

    def getContextVectors(self,input):
        '''

        '''
        values = self.W_v(input)
        attention_weights = self.getAttentionWeight(input)
        context_vectors = attention_weights @ values
        return context_vectors

    @staticmethod
    def test_getMockReturn():
        manual_seed(517)
        inputs = Tensor(
            [[0.43, 0.15, 0.89], # Your 
            [0.55, 0.87, 0.66], # journey
            [0.57, 0.85, 0.64], # starts
            [0.22, 0.58, 0.33], # with 
            [0.77, 0.25, 0.10], # one 
            [0.05, 0.80, 0.55]] # step
        )
        self_attentionv2 = SelfAttentionV2(3,6)
        context_vectors = self_attentionv2(inputs)
        logger.info(f'Return context_vectors Shape : {context_vectors.shape}')
        logger.info(f'Return context_vectors : {context_vectors}')
        return context_vectors

if __name__ == '__main__':
    SelfAttentionV2.test_getMockReturn()