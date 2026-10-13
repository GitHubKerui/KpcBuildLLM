from kpc_llm.utils import  getlogger
import torch

log = getlogger()

def prepare_finetuning_model(model):
    # log_model(model)
    # print(model)
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # (out_head): Linear(in_features=1280, out_features=50257, bias=False)
    # 替换最后一层 linear 因为是2 classes
    """ 这个种子非常重要，如果不加可能导致初始化数据有问题，导致梯度反向，训练崩溃。 """
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(1280,2)
    # 放开这倒数二层三层层
    # (trf_blocks): Sequential()
    # (final_norm): LayerNorm()
    for param in model.trf_blocks[-1].parameters():    
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    return model
