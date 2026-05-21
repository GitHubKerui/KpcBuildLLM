from turtle import forward
from torch import nn,softmax,manual_seed,Tensor
from kpc_llm.data_process import drop_out
from kpc_llm.utils.logger import getlogger
from kpc_llm.data_process.causal_attention_process import setUpNegativeInfMask


logger = getlogger()

class CausalSelfAttention(nn.Module):
    def __init__(self,in_d,out_d,drop_rate:float) -> None:
        super().__init__()
        self.W_q = nn.Linear(in_d,out_d)
        self.W_k = nn.Linear(in_d,out_d)
        self.W_v = nn.Linear(in_d,out_d)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self,input):
        queries = self.W_q(input)
        keys = self.W_k(input)
        values = self.W_v(input)
        attention_scores = queries @ keys.T
        causal_attention_scores = setUpNegativeInfMask(attention_scores,self)
        attention_weights = softmax(causal_attention_scores / queries.shape[-1]**0.5 , dim = -1)
        drop_out_attention_weights = self.dropout(attention_weights)
        logger.info(f'drop_out_attention_weights : {drop_out_attention_weights}')
        context_vectors = drop_out_attention_weights @ values
        logger.info(f'values : {values}')
        return context_vectors

if __name__ == '__main__':
    manual_seed(517)
    inputs = Tensor(
        [[0.43, 0.15, 0.89], # Your 
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with 
        [0.77, 0.25, 0.10], # one 
        [0.05, 0.80, 0.55]] # step
    )
    causalSelfAttention = CausalSelfAttention(3,16,0.5)
    context_vectoers = causalSelfAttention(inputs)
    logger.info(f'Return causalSelfAttention context_vectoers : {context_vectoers}')