
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
        queries = self.W_q(input)
        keys = self.W_k(input)
        Values = self.W_v(input)
        attention_scores = queries @ keys.T
        attention_weights = softmax(attention_scores / queries.shape[-1]**0.5 , dim = -1)
        context_vector = attention_weights @ Values
        return context_vector

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
    self_attentionv2 = SelfAttentionV2(3,5)
    context_vectors = self_attentionv2(inputs)
    logger.info(f'SelfAttentionV2 : {context_vectors}')