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
    # 新 head 默认建在 CPU，必须跟着主干走。否则前向会报
    # "Expected all tensors to be on the same device, but got mat1 is on cuda:0,
    #  different from other tensors on cpu"。
    # 以前靠调用方事后补一句 model.to(device) 兜住（手写循环 / Lightning 的 Trainer 都会补），
    # 但绕过这两条路的调用方（例如训练前做 baseline 自检）就会直接踩坑。
    model.out_head = torch.nn.Linear(1280, 2).to(next(model.parameters()).device)
    # 放开这倒数二层三层层
    # (trf_blocks): Sequential()
    # (final_norm): LayerNorm()
    for param in model.trf_blocks[-1].parameters():    
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    return model
