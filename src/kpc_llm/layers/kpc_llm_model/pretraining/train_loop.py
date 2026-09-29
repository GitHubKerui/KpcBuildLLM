# 1 加载训练集数据，加载验证集数据。
# 2 配置modal训练超参
# 3 训练modal，验证modal,测试modal评测得分
# 4 loop 1，2，3，直到modal 的benchmark得分理想。



from logging import Logger


from torch.utils.data import dataloader
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.train_loss_calcu import calcuOneBatchCrossEnLoss
from kpc_llm.layers.loss_func.kpc_cross_entropy import target_batch
from kpc_llm.utils.logger import getlogger
import torch

logger: Logger = getlogger()

def train_loop(epochNum:int,dataloader,model:KpcLLMModel,optimizer,device):
    # 一个epoch时代，纪元，是全量数据训练一次
    for i in range(epochNum):
        logger.info(i)
        for i,(trainBatch,target_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            # 每个批次进行propagate forward算出loss
            loss = calcuOneBatchCrossEnLoss(trainBatch,target_batch,model,device)
            # backward 算出梯度 ,算之前需要先设梯度为zere或者none
            loss.backward()
            # loss.backward后，所有计算图链条上的weight的gradient被计算出来，用optimizer的step进行计算更新。
            optimizer.step()
            


torch.manual_seed(825)
TRAIN_CNF = {

}
model = KpcLLMModel(TRAIN_CNF)
print (f"cuda is {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),0.001,weight_decay=0.1)

if __name__ == "__main__":

    return None
    # train_loop(9)