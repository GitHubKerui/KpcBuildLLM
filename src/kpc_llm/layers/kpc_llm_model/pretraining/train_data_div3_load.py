# 1 从文件中读取pretraining 数据。
# 2 把数据处理成两部分，第一部分，推理数据，第二部分 标签数据，组成一一对应的格式
# 3 把两部分数据划分成三个不同集合组，训练集，验证集，最终测试集 大概是 8：1：1，当数据量较少的时候可以考虑K折交叉验证
# 4 训练部分数据 需要tokenizer的处理。后进入模型训练。
from multiprocess.pool import worker

from kpc_llm.layers.kpc_llm_model.token_process.tokenid_ds_loader import create_dataloader
from kpc_llm.data_fetch.textloader import getTxtStr

import torch
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.train_loss_calcu import caluBatchesCrossEnLoss
import tiktoken 

# chunk_len 后面就是llm上下文的长度 ，batch_size是多组上下文训练数据一组
def divDatas2TraValTes(tokenids:torch.Tensor,batch_size,chunk_len,stride,shuffle=False,drop_last=False,num_worker=4):
    tokenids = tokenids.view(-1)
    dataLen = len(tokenids)
    trainLastInd = int(dataLen * 0.8)
    valLastInd = int (dataLen * 0.9)

    trainDataset = tokenids[:trainLastInd]
    valDataset = tokenids[trainLastInd:valLastInd]
    testDataset = tokenids[valLastInd:]
    
    trainDatasetloader = create_dataloader(trainDataset,batch_size,chunk_len,stride,shuffle,drop_last,num_worker)
    valDatasetloader = create_dataloader(valDataset,batch_size,chunk_len,stride,shuffle,drop_last,num_worker)
    testDatasetloader = create_dataloader(testDataset,batch_size,chunk_len,stride,shuffle,drop_last,num_worker)

    return trainDatasetloader,valDatasetloader,testDatasetloader

# TODO k折交叉处理 暂时不用在这里写。
def k_fold_cross_process(dataloader):
    return None


# for test
if __name__ == "__main__":
    # get TrainData from data/the-verdict.txt
    train_txt = getTxtStr('the-verdict.txt','data')
    train_txtln  = len(train_txt) if train_txt is not None else 0

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens_ids = tokenizer.encode(train_txt,allowed_special={'<|endoftext|>'})
    print(f"train_txt length : {train_txtln}")
    trainDsloader,valDsloader,testDsloader = divDatas2TraValTes(tokenids=torch.tensor(tokens_ids),batch_size=4,chunk_len=256,stride=256,num_worker=4)
    # ds1 = trainDsloader.dataset
    # sample_num = ds1.chunk_len 
    # batch_num = trainDsloader.batch_size
    # token_num = batch_num * sample_num
    # print(f"sample_num : {sample_num} batch_num : {batch_num} token_num : {token_num}")
    for train,target in trainDsloader:
        print(f"train shape :{train.shape}")
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
        totalLoss = caluBatchesCrossEnLoss(trainDsloader,kpc_llm_model,device,0)
    print(f" totalLoss_e :{totalLoss}")
    print(f" totalLoss_e perplexity :{torch.exp(torch.tensor(totalLoss))}")