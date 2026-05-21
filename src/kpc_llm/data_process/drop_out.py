from kpc_llm.layers import self_attention_v2
from kpc_llm.utils.logger import getlogger
from kpc_llm.layers.self_attention_v2 import SelfAttentionV2

from torch import Tensor,nn,manual_seed

def add_drop_mask(input):
    '''
    
    '''
    mask = getDropOutMask()
    masked_input = None
    return masked_input

def getDropOutMask():
    '''
    '''
    mask = None
    return mask

if __name__ =='__main__':
    manual_seed(517)
    test_input = Tensor([
        [0.43, 0.15, 0.89], # Your 
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with 
        [0.77, 0.25, 0.10], # one 
        [0.05, 0.80, 0.55]  # step
        ])
    selfAttentionV2 = SelfAttentionV2(3,6)
    attention_scores = selfAttentionV2.getAttentionScores(test_input)