from torch import Tensor,nn,manual_seed,ones,tril,triu,inf,bool
from kpc_llm.utils.logger import getlogger

logger = getlogger()

def setUpZeroMask(attention_scores,module:nn.Module):
    '''
    目标：使得矩阵上三角部分为0，下三角部分不变，
    通过下三角全1把对角线上半部设置为0来与非计算，遮盖上三角部分的注意力分数，也就是上三角归零计算
    自注意力分数是词元和上下文的关系（方阵）,最后是一定是个方阵，方阵维度等于上下文的--“词元长度”。
    '''
    context_len = attention_scores.shape[-2]
    onesMatrix = ones(context_len,context_len,device=attention_scores.device)
    #triangle + lower =tril 下三角包括对角线位置的1保留，其他设置为0
    trilOneMask = tril(onesMatrix)
    module.register_buffer("trilOneMask",trilOneMask)
    masked_attention_scores = attention_scores * module.trilOneMask
    return masked_attention_scores

def setUpNegativeInfMask(attention_scores,module:nn.Module):
    '''
    通过上三角全1把负无穷矩阵对角线上半部设置为全部负无穷，下半部为0，然后通过加法计算，遮盖上三角部分的注意力分数为负无穷，也就是上半部负无穷设置
    '''
    #因为注意力的mask是二维的，所以一定是以倒数第二个维度来做的，最后一个维度是单个词元的嵌入空间维度。
    context_len = attention_scores.shape[-2]
    # 更正命名：这是一个全 1 矩阵（推荐使用 torch.ones，并指定设备device和输入一致，以防报错）
    # 不设置会报错 expected self and mask to be on the same device, but got mask on cpu and self on cuda:0
    onesMatrix = ones(context_len,context_len,device=attention_scores.device)
    #triangle + upper = tri + up = triu 上三角不包括对角线设置1的方法,diagonal = 1 是指的包含的对角线往右移动
    triuNegativeInfinityMask = triu(onesMatrix, diagonal = 1)
    # 注册为非持久缓冲层（Buffer 会随模型一同移动到 GPU，且不作为训练参数）
    module.register_buffer("mask",triuNegativeInfinityMask)
    # 将掩码为 1 的地方填充为负无穷
    attention_scores_masked = attention_scores.masked_fill(module.mask.bool(),-inf)
    return attention_scores_masked    

def getTriuTrueMask(context_len):
    '''
    通过上三角全1把负无穷矩阵对角线上半部设置为全部负无穷，下半部为0，然后通过加法计算，遮盖上三角部分的注意力分数为负无穷，也就是上半部负无穷设置
    '''
    # 更正命名：这是一个全 1 矩阵（推荐使用 torch.ones，并指定设备device和输入一致，以防报错）
    # 不设置会报错 expected self and mask to be on the same device, but got mask on cpu and self on cuda:0
    # 这里在attentionblock的init模块里register会随device调入cuda，不用再设置device
    onesMatrix = ones(context_len,context_len,dtype=bool)
    #triangle + upper = tri + up = triu 上三角不包括对角线设置1的方法,diagonal = 1 是指的包含的对角线往右移动
    triuNegativeInfinityMask = triu(onesMatrix, diagonal = 1)
    return triuNegativeInfinityMask
    # 注册为非缓冲层（Buffer 会随模型一同移动到 GPU，且不作为训练参数，需要在__init__中调用）
    # module.register_buffer("causal_mask",triuNegativeInfinityMask,persistent=False)
 

if __name__ =='__main__':

    from kpc_llm.layers.kpc_llm_model.test.algorithm_test.self_attention_v2 import SelfAttentionV2

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