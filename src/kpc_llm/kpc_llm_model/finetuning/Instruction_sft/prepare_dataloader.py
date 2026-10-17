

import json
from pathlib import Path

from pandas.core.frame import DataFrame
import pandas as pd
import torch
from kpc_llm.kpc_llm_model.finetuning.classify_sft.SpamDataset import SpamDataset
from kpc_llm.kpc_llm_model.finetuning.Instruction_sft.SftDataset import SftDataset
from kpc_llm.layers.loss_func.kpc_cross_entropy import target_batch
from kpc_llm.utils import download_file, getlogger,get_project_root
import tiktoken
from torch.utils.data import DataLoader, Subset
# 偏函数（Partial Function） 是一个来自数学和函数式编程的概念。
# 简单来说，它的核心作用就是“参数预设”：把一个有很多参数的函数，
# 提前固定住其中几个参数，从而创造出一个参数更少、用起来更简单的新函数
from functools import partial

from kpc_llm.utils.adaptive_device import get_adaptive_device

tokenizer = tiktoken.get_encoding("gpt2")


log =  getlogger()

# 反面教材留着
def div_df(json_dt):
    """ 切数据 dataset 不能直接切索引，因为 SftDataset 返回 token_ids, prompt_len，会触发重新 tokenize"""
    train_tail = int(len(json_dt)*0.85)
    val_tail = int(len(json_dt)*0.9)

    train_json = json_dt[:train_tail]
    val_json = json_dt[train_tail:val_tail]
    test_json = json_dt[val_tail:]
    log.info(f"\n train_json: {len(train_json)} | val_json : {len(val_json)} | test_json : {len(test_json)}")
    return train_json,val_json,test_json

def custom_collate_fn(batch,pad_token_id =50256,ignore_index=-100,allowed_max_length=1024,device="cpu"):
    """ 把一个 batch 的 (token_ids, prompt_len) 补齐成等长，切出 inputs / targets。

    - inputs 取 [:-1]、targets 取 [1:]：标准的左移一位，做 next-token prediction
    - prompt 段的标签置 ignore_index，只对 response 算 loss
    - padding 段也置 ignore_index，但每条保留一个 pad_token_id 作为「回复结束」信号
    """
    # 截断要放在算长度之前：否则 max_len 会被未截断的长样本顶高，整批都被迫变长
    samples = batch if allowed_max_length is None else [
        (item[:allowed_max_length], min(prompt_len, allowed_max_length))
        for item, prompt_len in batch
    ]
    # +1 是为了左移：inputs 取 [:-1]、targets 取 [1:]，两者长度才一致
    batch_max_length = max(len(item) for item, _ in samples) + 1

    inputs_lst, targets_lst = [], []
    for item, prompt_len in samples:
        # 输入右侧补 pad_token_id
        padded_inputs = item + [pad_token_id] * (batch_max_length - len(item))
        # 标签：内容后先补一个真实的 pad_token_id 作为结束信号，其余 padding 补 ignore_index
        padded_targets = item + [pad_token_id] + [ignore_index] * (batch_max_length - len(item) - 1)
        # prompt 段整体屏蔽；prompt_len 正常情况下不会超过 len(item)，截断后夹一下保险
        mask_len = min(prompt_len, len(item))
        padded_targets[0:mask_len] = [ignore_index] * mask_len

        inputs_lst.append(torch.tensor(padded_inputs[:-1]))
        targets_lst.append(torch.tensor(padded_targets[1:]))

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def prepare_dataloader(batch_size:int=8,num_worker:int=0):

    torch.manual_seed(123)
    dataset = SftDataset()

    # 不能用 div_df(dataset) 切分：SftDataset.__getitem__ 返回 (token_ids, prompt_len)，
    # 对它 slice 会走 Python 序列协议，拿到 (全部token, 全部prompt_len) 这个二元组，
    # len() 恒为 2，切分比例全废。Subset 只传索引，也不会触发重新 tokenize。
    total = len(dataset)
    train_tail, val_tail = int(total*0.85), int(total*0.9)
    train_dataset = Subset(dataset, range(0, train_tail))
    val_dataset = Subset(dataset, range(train_tail, val_tail))
    test_dataset = Subset(dataset, range(val_tail, total))

    # 必须显式指定 collate_fn：默认的 default_collate 处理不了长短不一的 token id 列表，
    # 会抛 "each element in list of batch should be of equal size"
    loader_common = dict(
        num_workers=num_worker,
        collate_fn=custom_collate_fn,
        # num_workers=0 时写 True 会抛 ValueError: persistent_workers option needs num_workers > 0
        persistent_workers=num_worker > 0,
    )
    train_loader = DataLoader(train_dataset,batch_size,True,drop_last=True,**loader_common)
    val_loader = DataLoader(val_dataset,batch_size,drop_last=False,**loader_common)
    test_loader = DataLoader(test_dataset,batch_size,drop_last=False,**loader_common)
    log.info(f"\n train_loader: {len(train_loader)}")
    log.info(f"\n val_loader  : {len(val_loader)}")
    log.info(f"\n test_loader : {len(test_loader)}")
    return train_loader,val_loader,test_loader


if "__main__" == __name__:
    # 先单步验证 collate_fn 的 mask 是否符合预期，再跑完整 DataLoader
    dataset = SftDataset()
    first_ids, first_prompt_len = dataset[0]
    print(f"首条: token 数 {len(first_ids)} | prompt token 数 {first_prompt_len}")

    # 拿 3 条当一批；别把整个 dataset 当 batch，那会把所有样本 pad 到全数据集最长
    inputs, targets = custom_collate_fn([dataset[0], dataset[1], dataset[2]], device="cpu")
    print(f"shapes: {tuple(inputs.shape)} {tuple(targets.shape)}")
    print(f"inputs0: {inputs[0]} | targets0 {targets[0]}")

    row = 0
    print(f"prompt 段已屏蔽(-100): {(targets[row, :first_prompt_len - 1] == -100).all().item()}")
    print(f"首个 response token 保留: {targets[row, first_prompt_len - 1].item()} (不应等于 -100)")
    print(f"response 段含有效标签: {(targets[row, first_prompt_len - 1:] != -100).any().item()}")
    print(f"labels 里混入真实 pad(50256) 的次数: {(targets == 50256).sum().item()} (应等于批大小，即每条一个结束信号)")

    # 上面都符合预期后，再打开这行跑完整 DataLoader
    # train_loader, val_loader, test_loader = prepare_dataloader()
    # for step, (inputs, targets) in enumerate(train_loader):
    #     print(step, tuple(inputs.shape), tuple(targets.shape))