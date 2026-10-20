

from typing import Any
from torch._tensor import Tensor
import torch
from peft import LoraConfig,get_peft_model

from kpc_llm.utils.adaptive_device import get_adaptive_device


class FreezeModel(torch.nn.Module):
    """ 冻结基座模型：参数的 requires_grad=False  """
    def __init__(self,lora_model):
        super().__init__()
        self.model = lora_model
        
        # 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # 获取原始模型状态，放置val或test阶段导致后面的train把状态开错，只有真training 下面的 train(才会生效)
        was_training = self.model.training
        self.model.eval()
        x = self.model(x)
        self.model.train(was_training)
        return x



class LoraLayer(torch.nn.Module):
    """LoRA：out = base(x) + scaling·B(A(x))，scaling=alpha/rank，只有 A/B 可训练"""
    def __init__(self,lora_model,in_dim,rank,out_dim,alpha=None):
        super().__init__()
        self.freeze_model = FreezeModel(lora_model)
        self.lora_metrix_a = torch.nn.Linear(in_dim, rank,bias=False)
        self.lora_metrix_b = torch.nn.Linear(rank, out_dim,bias=False)
        # A 随机初始化、B 零初始化 → 训练起点 ΔW=B·A=0，首次前向不改变基座输出，第一次 loss 就是基座算出来的
        torch.nn.init.zeros_(self.lora_metrix_b.weight)
        # 分类头
        self.modelTarget = torch.nn.Linear(out_dim, 2)
        # loRA缩放系数
        self.scaling  =   (alpha if alpha is not None else rank)/ rank

    def forward(self, x):
        base_x  = self.freeze_model(x)
        ax = self.lora_metrix_a(x)
        lora_x = self.lora_metrix_b(ax)
        return base_x + lora_x * self.scaling

def make_data(n=128,in_dim = 6,device:torch.device | str | None = None):
    """ 
    随机制造样本点 和 标签
    n 代表样本数，也就是说这是一个batch的数据
    in_dim 代表样本的相关特征组合
    w的out相当于 特征投影后的分类头数
    argmax 等于把投影后最大分量当targe
    """
    # 定随机种，保证结果可浮现
    torch.manual_seed(123)
    x: Tensor = torch.randn(n,in_dim,device=device or "cpu")
    w: Tensor = torch.randn(in_dim,2,device=device)
    y: Any = (x @ w).argmax(-1)
    input: Tensor = x
    target: Any = y
    return input,target

def test_self_lora():
    """ 测试Lora效果 """
    device = get_adaptive_device()
    # 造数据
    input, target = make_data(256, 6,device=device)
    base = torch.nn.Linear(6, 6)
    model = LoraLayer(base,6,2,6).to(device)
    # 把冻结基座模型的参数分离，拷贝一份出来
    base_w = base.weight.detach().clone()
    print_base_lora(model)
    mock_train_and_compare(input,model,target,base,base_w)

def test_peft_lora_compare():
    """ 测试参数效率微调框架 peft和自己写的是否有区别 """
    device = get_adaptive_device()
    torch.manual_seed(123)
    base_ours  = torch.nn.Linear(6,6)
    ours_lora_model = LoraLayer(base_ours,6,2,6,4).to(device) # scaling = 4/2 =2
    base_ours_w = base_ours.weight.detach().clone()            # .to(device) 之后再拷贝，否则 torch.equal 会设备不一致
    # peft 靠 target_modules 按模块名注入，裸 Linear 没有子模块名，用 Sequential 包一层
    base_peft = torch.nn.Linear(6, 6)
    peft_lora_model = get_peft_model(
        torch.nn.Sequential(base_peft),
        LoraConfig(r=2,lora_alpha=4,target_modules=["0"])
    ).to(device)

    # 为了比较先重新设置会非0
    torch.nn.init.normal_(ours_lora_model.lora_metrix_b.weight)

    for name,p in peft_lora_model.named_parameters():
        if "lora_A" in name:
            p.data.copy_(ours_lora_model.lora_metrix_a.weight.data)
        elif "lora_B" in name:
            p.data.copy_(ours_lora_model.lora_metrix_b.weight.data)
        elif name.endswith("weight"):
            p.data.copy_(base_ours.weight.data)
        elif name.endswith("bias"):
            p.data.copy_(base_ours.bias)

    # 基准必须在对齐之后记录：上面把 base_ours 的权重拷进了 peft 的 base_layer
    base_peft_w = base_peft.weight.detach().clone()

    # 造数据
    input, target = make_data(256, 6,device=device)
    print("compare--------------------test_peft_lora_compare---start----------------\n")
    mock_train_and_compare(input,ours_lora_model,target,base_ours,base_ours_w)
    print("compare-----------------------start----------------\n")
    mock_train_and_compare(input,peft_lora_model,target,base_peft,base_peft_w)
    print("compare finish\n")

    

def print_base_lora(model):
    # 打印基座模型 和 lora层的参数信息
    train_p = [(n,p) for n,p in model.named_parameters() if p.requires_grad ]
    lora_p =  [ (n,p) for n,p in train_p if "lora_metrix" in n]
    train_p_n = sum([ p.numel() for n,p in train_p])
    lora_p_n = sum([ p.numel() for n,p in lora_p])
    print(f"trainable params: {train_p_n}  (LoRA {lora_p_n} + 分类头 {train_p_n - lora_p_n})")

def mock_train_and_compare(input,model,target,base_model,base_w):
    trainable_p = [p for p in model.parameters() if p.requires_grad]
    opitimizer = torch.optim.AdamW(trainable_p,lr=8e-3)
    # 单批次 跑200epoch
    for step in range(900):
        predict = model(input)
        loss = torch.nn.functional.cross_entropy(predict,target)
        opitimizer.zero_grad()
        loss.backward()
        opitimizer.step()
        # 类似val的打印
        if step % 50 == 0:
            print(f"step {step:3d} loss {loss.item():.4f}")
    print(f"final loss {loss.item():.4f}")
    # 必须是（assert） loss.item()<0.1, 否则 raise exception
    assert loss.item() < 0.1 , "你学废了吗？哈哈"
    # 必须是 训练后的基座权重 和 训练前的copy是一样 否则 给老子崩
    assert torch.equal(base_model.weight.detach(),base_w),"基座权重不可能变了吧，我草"
    # all assert pass then
    print(f"最终loss也收敛到了 0.1以下，学习正常。")
    print("基座模型的weight没有变化， 说明loRA是OK的。\n")

if __name__ == "__main__":
    test_self_lora()
    test_peft_lora_compare()