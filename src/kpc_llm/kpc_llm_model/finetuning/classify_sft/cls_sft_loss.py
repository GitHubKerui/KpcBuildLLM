
import random

from torch.nn.functional import cross_entropy

from kpc_llm.utils.logger import getlogger


log =  getlogger()

""" cross_entropy 
        **二标标准输入格式**

        标准二维模式：input（logits）：形状为 [N, C]，其中 N 是样本总数，C 是类别总数（即 vocab_size）。
        target：形状为 [N]，包含对应的类别索引标签。
        
        **高维模式**（如果输入是 3D）：
        * input：必须是 [N, C, d1, d2, ...]（注意：类别数 C 必须排在第 1 维 **）。
          target：形状为  [ C, d1, d2, ...]，包含对应的类别索引标签。
          target 的0维 必须跟 input的 1维 对齐
        所以这里是超过高维模式的情况 
        last_token_logit shape 肯定是     [batch_size , 1 , 2]
        target_batch     shape 肯定是                  [1]
    """
def one_batch_loss(train_batch,target_batch,model,device):
    train_batch = train_batch.to(device)
    target_batch = target_batch.to(device)
    logits_batch = model(train_batch)
    """ 分类微调的loss 计算的关键点,只把分类特征放到最后一个token里 """
    """ 因为因果注意力的最后字符是有全部上下文向量的信息的，特征最全, """
    last_token_logit = logits_batch[:,-1,:]
    # """ 这里last_token_logit shape肯定是 [batch_size,1,2] """
    # log.info(f"last_token_logit shape ：{last_token_logit.shape}")
    # """ 这里target_batch shape 肯定是 [1] """
    # log.info(f"target_batch shape ：{target_batch.shape}")
    loss = cross_entropy(last_token_logit,target_batch)
    return loss


def multi_batches_loss(dataLoader,model,device,batchNum=0):
    """ 计算多个batch的loss，但是只计算前batchNum个batch的loss，并不随机 """
    # 判断数据是否完整
    if dataLoader is None or  (dataLoader) == 0:
        return float('nan')
    elif batchNum is None or batchNum <= 0:
        # 如果没有指定计算前多少个batch的loss 则算全部的
        batchNum = len(dataLoader)
    else:
        realLen = len(dataLoader)
        # 如果指定了则取长度和指定数量的最小值。
        batchNum = min(batchNum,realLen)
    
    total_loss=0.
    for i,(train_batch,target_batch) in enumerate(dataLoader):
        if i < batchNum:
            loss = one_batch_loss(train_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break

    return total_loss / batchNum

def multi_batches_random_loss(dataLoader,model,device,batchNum=0):
    """ 解决多batch的随机抽取问题，保证评估分布质量 """
    # 判断数据是否完整
    if dataLoader is None or  (dataLoader) == 0:
        return float('nan')
    elif batchNum is None or batchNum <= 0:
        # 如果没有指定计算前多少个batch的loss 则算全部的
        realLen = len(dataLoader)
        batchNum = len(dataLoader)
    else:
        realLen = len(dataLoader)
        # 如果指定了则取长度和指定数量的最小值。
        batchNum = min(batchNum,realLen)

    # 随机抽取batchNum个batch索引,set的设置保证target_id是Set，查找哈希类型可以保证大数据的高效  # 
    target_ids = set(random.sample(range(realLen),batchNum))

    total_loss=0.
    for i,(train_batch,target_batch) in enumerate(dataLoader):
        """ 这里的target_id 是Set类型，in 瞬间完成：哈希查找，速度极快！"""
        if i in target_ids:
            loss = one_batch_loss(train_batch,target_batch,model,device)
            total_loss += loss.item()

    return total_loss / batchNum