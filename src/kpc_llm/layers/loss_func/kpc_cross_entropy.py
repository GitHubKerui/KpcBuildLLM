""" 
cross entropy交叉熵损失函数的基本原理：
    是用来比较两个分布差异的量化指标，这个函数可以用来做损失函数，从而计算梯度矩阵，反向优化模型。
    离散有限分类的情况，事件是有限的，所有事件预测的概率 和 真实目标概率的差值的均值 loss，的 对数函数，这样可以把很小的ch

这里是进行部分实验的code
"""
from torch import tensor,no_grad,softmax,argmax,index_select,gather,log,mean
from torch.nn.functional import cross_entropy
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel

# step 1
# here we have two batch data,one is training batch with shape [1,2,3]
# and another is target batch which has a shape [1,2,3] 
# last dim is token id
train_batch = tensor([[16833, 3626, 6100],   # "every effort moves",
                                   [40,    1107, 588 ]])               # "I really like"
target_batch =tensor([[3626, 6100, 345  ],   # " effort moves you",
                                   [1107,  588, 11311]])             # " really like chocolate"


# step 2 
# we have to get the prediction of the vocab distribution batch
# and then get the expection probability that target token have 
# and then get the error between the prediction and expection,actually is 1.

# 写配置
LLM_CONFIG = {
    "vcab_sz" : 50524,
    "cntext_lnth" : 3,
    # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
    "max_cntxt_pstion_lnth" : 3,
    "emb_dim" : 768,
    "heads_num" : 3,
    "trnsf_blocks_num" : 3,
    "drop_rt" : 0.1,
    "qkv_bias" : False,
    # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
    "returnSoftmax" : False
}

model = KpcLLMModel(LLM_CONFIG)
with no_grad():
    prediction_batch = model(train_batch)

pred_prob = softmax(prediction_batch,-1)
# 实际贪心算法中最高概率预测的tokenid
pred_tokenid = argmax(pred_prob,-1)
# 目标tokenid预测的真实概率 提取
# 获取pred_prob的shape
batch_num,token_num,vocab_num = pred_prob.shape
# target没有 这个 vocab的分布概率最后一维度，但是他的最后一维的值是对应的需要预测的token的id
# 为了最后处理，需要把最后一维度扩展出新维度和pred_prob的shape对齐，这里用squeeze挤出一个维度。
target_batch_sq = target_batch.unsqueeze(-1) 
# 沿最后一维利用收集gather概率
pred_target_prob = gather(pred_prob,-1,target_batch_sq)
# 再压平最后一个维度
pred_target_prob_flat = pred_target_prob.squeeze(-1)
# 用对数函数log把这些极小值变成负值，负值的数值相对较大，利于计算。这些负值的绝对值就是困惑度Perplexity
pred_perplexity = log(pred_target_prob_flat)*-1
# 最后计算批次平均困惑度 可以作为loss
pred_perplexity_mean = mean(pred_perplexity,-1)


print(f"---------------------------- ::::::::::::::::::::-----------------------------------")
print(f"prediction_batch shape : {prediction_batch.shape}")
print(f"prediction_batch : {prediction_batch}")
print(f"pred_prob shape : {pred_prob.shape}")
print(f"pred_prob : {pred_prob}")
print(f"pred_tokenid shape : {pred_tokenid.shape}")
print(f"pred_tokenid : {pred_tokenid}")
print(f"target_batch_sq shape : {target_batch_sq.shape}")
print(f"target_batch_sq : {target_batch_sq}")
print(f"pred_target_prob shape : {pred_target_prob.shape}")
print(f"pred_target_prob : {pred_target_prob}")
print(f"pred_target_prob_flat shape : {pred_target_prob_flat.shape}")
print(f"pred_target_prob_flat : {pred_target_prob_flat}")
print(f"pred_perplexity shape : {pred_perplexity.shape}")
print(f"pred_perplexity : {pred_perplexity}")
print(f"pred_perplexity_mean shape : {pred_perplexity_mean.shape}")
print(f"pred_perplexity_mean : {pred_perplexity_mean}")

