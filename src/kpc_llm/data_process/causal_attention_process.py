from torch._refs import zero
from kpc_llm.utils.logger import getlogger
from kpc_llm.layers.self_attention_v2 import SelfAttentionV2
from torch import Tensor,nn,manual_seed,ones,tril,triu,inf
from kpc_llm.utils.logger import getlogger

logger = getlogger()

def setUpZeroMask(attention_scores,module:nn.Module):
    '''
    通过下三角全1把对角线上半部设置为0来与非计算，遮盖上三角部分的注意力分数，也就是上三角归零计算
    '''
    context_len = attention_scores.shape[0]
    oneMatrix = ones(context_len,context_len)
    #triangle + lower =tril 下三角包括对角线保留其他设置为0
    trilOneMask = tril(oneMatrix)
    module.register_buffer("trilOneMask",trilOneMask)
    masked_attention_scores = attention_scores * module.trilOneMask
    return masked_attention_scores

def setUpNegativeInfMask(attention_scores,module:nn.Module):
    '''
    通过上三角全1把负无穷矩阵对角线上半部设置为全部负无穷，下半部为0，然后通过加法计算，遮盖上三角部分的注意力分数为负无穷，也就是上半部负无穷设置
    '''
    context_len = attention_scores.shape[0]
    zeroMatrix = ones(context_len,context_len)
    #triangle + upper = triu 上三角不包括对角线设置1的方法,diagonal = 1 是指的包含的对角线往右移动
    triuNegativeInfinityMask = triu(zeroMatrix, diagonal = 1)
    module.register_buffer("triuNegativeInfinityMask",triuNegativeInfinityMask)
    attention_scores_masked = attention_scores.masked_fill(module.triuNegativeInfinityMask.bool(),-inf)
    return attention_scores_masked    

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
    selfAttentionV2 = SelfAttentionV2(3,8)
    attention_scores = selfAttentionV2.getAttentionScores(test_input)
    attention_weights = selfAttentionV2.getAttentionWeight(test_input)
    context_vectoers = selfAttentionV2.getContextVectors(test_input)
    #和token数一样的矩形
    logger.info(f' attention_scores shape : {attention_scores.shape}')
    logger.info(f' attention_weights shape : {attention_weights.shape}')
    #context_vectoers 有和token数一样的shape[0] 和q k v的输出维度一样的 shape[-1]，
    #qkv的 dim[-1] 决定了token上下文向量的空间维度。
    logger.info(f' context_vectoers shape : {context_vectoers.shape}')

    zero_masked_scores = setUpZeroMask(attention_scores,selfAttentionV2)
    logger.info(f' zero_masked_scores : {zero_masked_scores}')
    neg_inf_masked_scores = setUpNegativeInfMask(attention_scores,selfAttentionV2)
    logger.info(f' neg_inf_masked_scores : {neg_inf_masked_scores}')