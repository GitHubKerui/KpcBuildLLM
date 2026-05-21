from kpc_llm.layers import self_attention_v2
from kpc_llm.utils.logger import getlogger
from kpc_llm.layers.self_attention_v2 import SelfAttentionV2
from torch import Tensor,nn,manual_seed

logger = getlogger()
def add_drop_out(attention_weights,dropRate):
    '''
    在注意力的权重计算后加上drop_out来减少训练过程中的overfitting，一般比例设置0.1-0.5
    在全one矩阵的基础上随机drop位置被归零，留存下来的1会 (1 * 1/dropRate)，最后和原始矩阵相乘，不是dot乘
    '''
    dropoutMask= nn.Dropout(0.5)
    dropOutWeights = dropoutMask(attention_weights)
    return dropOutWeights

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
    attention_weights = selfAttentionV2.getAttentionWeight(test_input)
    attention_weights_drop = add_drop_out(attention_weights,0.5)
    logger.info(f'attention_weights : {attention_weights}')
    logger.info(f'attention_weights sum : {attention_weights.sum(dim=-1)}')
    logger.info(f'attention_weights_drop : {attention_weights_drop}')
    logger.info(f'attention_weights_drop sum : {attention_weights_drop.sum(dim=-1)}')