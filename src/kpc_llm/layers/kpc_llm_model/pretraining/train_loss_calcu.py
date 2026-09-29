"""
计算预训练损失模块 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
import  torch
import math
from torch.utils.data import dataloader
from kpc_llm.data_fetch.textloader import getTxtStr
from kpc_llm.layers.kpc_llm_model.token_process.tokenid_ds_loader import create_dataloader
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.utils.logger import getlogger

logger = getlogger()
#1 因为损失是按照batch计算平均值的。所以要基本的 batch平均loss的计算。
def calcuOneBatchCrossEnLoss(train_batch:torch.Tensor,target_batch:torch.Tensor,model,device):
    # 将数据加载到CUDA里
    train_batch,target_batch = train_batch.to(device),target_batch.to(device)
    # 模型向前传播 (training使用)
    predict_train:torch.Tensor= model(train_batch)
    # 计算交叉熵损失(input is unnormalized logits) 输入是逻辑值，不是softmax的概率值
    crossEnLoss = torch.nn.functional.cross_entropy(predict_train.flatten(0,1),target_batch.flatten())
    return crossEnLoss

def caluBatchesCrossEnLoss(dataLoader,model,device,batchNum=0):
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
            loss = calcuOneBatchCrossEnLoss(train_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break

    return total_loss / batchNum

def test():
      # get TrainData from data/the-verdict.txt
    train_txt = getTxtStr('the-verdict.txt','data')
    logger.info(f"train_txt length : {len(train_txt)}")
    dataloader = create_dataloader(train_txt,4,512,512)
    # 写配置
    LLM_CONFIG = {
        "vcab_sz" : 50524,
        "cntext_lnth" : 512,
        # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
        "max_cntxt_pstion_lnth" : 512,
        "emb_dim" : 512,
        "heads_num" : 8,
        "trnsf_blocks_num" : 8,
        "drop_rt" : 0.1,
        "qkv_bias" : False,
        # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
        "returnSoftmax" : False
    }
    torch.manual_seed(825)
  
    with torch.no_grad():
        # 实例化LLM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kpc_llm_model = KpcLLMModel(LLM_CONFIG).to(device)
        totalLoss = caluBatchesCrossEnLoss(dataloader,kpc_llm_model,device,0)
    print(f" totalLoss_e :{totalLoss}")
    print(f" totalLoss_e perplexity :{torch.exp(torch.tensor(totalLoss))}")
    # PPL以2为底算出的困惑度更准，换底测试下。换底公式 log2(x) =  lnx/ln2 ,默认的corssentropy内部用e作为底。所以不能进行这个测试。
    # 问题出在初始权重上的负优化了。
    # totalLoss_2 = torch.tensor(totalLoss)/math.log(2)
    # print(f" totalLoss_2 :{totalLoss_2}")
    # totalLoss_2 = torch.pow(2,totalLoss_2)
    # print(f" totalLoss_2 perplexity :{torch.exp(torch.tensor(totalLoss_2))}")

    #1.2 TODO MSE/MAE均方差损失函数


# for test
if __name__ == "__main__":
    test()