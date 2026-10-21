
import itertools
import pprint
from functools import partial
from typing import Any
import torch
from torch.utils.data import DataLoader
from kpc_llm.kpc_llm_model.finetuning.direct_preference_sft.dpo_dataset import PreferDataset, data_path, div_data, load_data
from kpc_llm.kpc_llm_model.finetuning.train_main_loop import tokenizer
from kpc_llm.utils.adaptive_device import get_adaptive_device

def collate_fn(batch,pad_token_id=50256,allowed_max_length=None,mask_prompt_tokens=True,device:Any=torch.device("cpu")):
    # 初始化批次数据list用来装batch data
    batch_data: dict[str, Any[Any]] ={
        "prompt":[],
        "chosen":[],
        "rejected":[],
        "rejected_mask":[],
        "chosen_mask":[]
    }

    # 用最长的序列长度设置padding的长度
    max_length = 0
    if batch:
        for key in ["chosen","rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length = max(max_length,current_max)

    # 处理每个batch，
    for item in batch:
        # tokenid 转成 tensor 装进去
        prompt = torch.tensor(item["prompt"])
        prompt_len = prompt.shape[0]
        batch_data["prompt"].append(prompt)

        for key in ["chosen","rejected"]:
            # 根据最大长度padding
            seq = item[key]
            # 注意这里的pad的id是加了中括号的
            seq_pd = seq + [pad_token_id] * (max_length - len(seq))

            # 初始化全长度的为1的tensor的mask，1是true
            seq_mask = torch.ones(len(seq_pd)).bool()
            # seq_pd_mask的padding都设置成False 可以试试0
            seq_mask[len(seq):] = False
            # 设置提示词区域为False，+2 是因为有两个\n
            if mask_prompt_tokens:
                seq_mask[:prompt_len+2] = False
            # 装padding后的tokens的tensor 和 mask掩码tensor序列
            batch_data[key].append(torch.tensor(seq_pd))
            batch_data[f"{key}_mask"].append(seq_mask)
    
    # 最后的处理
    for key in  ["chosen","rejected","chosen_mask","rejected_mask"]:
        # 所有seq 目前还是list 需要全部stack 成tensor
        tensor_seq = torch.stack(batch_data[key])
        # 如果有context_lenth的限制则需要在最后一个维度截断
        if allowed_max_length is not None:
            tensor_seq = tensor_seq[...,:allowed_max_length]
        # 移动到指定的device
        batch_data[key] = tensor_seq.to(device)

    return batch_data

device  = get_adaptive_device()
customized_collate_fn = partial(collate_fn,device=device,mask_prompt_tokens=True,allowed_max_length=1024)

# 检查用
def decode_tokens_from_batch(token_ids, tokenizer):
    # print(f"token_ids : {token_ids}")
    ids_in_python_list = token_ids.flatten().tolist()
    # print(f"ids_in_python_list : {ids_in_python_list}")
    return tokenizer.decode(ids_in_python_list)

# dataloader查看用
def dataloader_slice(data):
    data = load_data(data_path)
    # pprint.pp(data[:2])
    example_dataset = PreferDataset(data)
    dataloader = DataLoader(example_dataset,
    batch_size=2,
    collate_fn=customized_collate_fn,
    shuffle=False)
    # 取出索引为 10 到 14 的 Batch（即第 11 到第 15 个 Batch，共 5 个）
    start_idx = 0
    end_idx = 3
    # islice 会在后台自动跳过前 10 个，然后把接下来的 5 个吐给你，完事立刻停止
    middle_batches = itertools.islice(dataloader, start_idx, end_idx)
    middle_batches_2 = itertools.islice(dataloader, start_idx, end_idx)
    for idx,batch in enumerate(middle_batches):
        # print("first_batch keys:", batch.keys())
        # [0] for the first entry in the batch
        text = decode_tokens_from_batch(token_ids=batch["prompt"][0],tokenizer=tokenizer)
        print(f"idx:{idx}--------------------------------\n\n{text}")

    for idx,batch in enumerate(middle_batches_2):
        # print("first_batch keys:", batch.keys())
        # [0] for the first entry in the batch
        text = decode_tokens_from_batch(token_ids=batch["rejected"][0],tokenizer=tokenizer)
        print(f"re idx:{idx}--------------------------------\n\n{text}")

def yeild_div_dataloader(data):
    num_workers = 0
    batch_size = 8
    data = load_data(data_path)
    train_data, test_data, val_data =  div_data(data)
    
    train_dataset = PreferDataset(train_data)
    train_dataloader = DataLoader(train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    drop_last=True,
    num_workers=num_workers,
    shuffle=True)

    test_dataset = PreferDataset(data=test_data)
    test_dataloader = DataLoader(test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    drop_last=False,
    num_workers=num_workers,
    shuffle=False)

    val_dataset = PreferDataset(data=val_data)
    val_dataloader = DataLoader(val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    num_workers=num_workers,
    drop_last=False,
    shuffle=False)

    return train_dataloader,test_dataloader,val_dataloader
        

def test():
    """ dict_keys(['prompt', 'chosen', 'rejected', 'rejected_mask', 'chosen_mask']) """
    data = load_data(data_path)
    # # pprint.pp(data[:2])
    # example_dataset = PreferDataset(data)
    # test_dataloader = DataLoader(example_dataset,
    # batch_size=2,
    # collate_fn=customized_collate_fn,
    # shuffle=False)

    # first_batch = next(iter(test_dataloader))
    # text = decode_tokens_from_batch(token_ids=first_batch["prompt"][0],  # [0] for the first entry in the batch
    #     tokenizer=tokenizer,
    # )
    # print("first_batch keys:", first_batch.keys())
    # print("text:\n\n", text)
    # print("first_batch shape:", first_batch["chosen"].shape)
    print("Train loader:")
    train_dataloader,test_dataloader,val_dataloader = yeild_div_dataloader(data)
    first_batch = next(iter(train_dataloader))
    print(first_batch.keys())
    print(f"\nchosen :\n{decode_tokens_from_batch(first_batch["chosen"][0],tokenizer=tokenizer)}")
    print(f"\nchosen_mask :\n{first_batch["chosen_mask"][0]}")
    # 先按 mask 筛出 response 段的 token id，再 decode；对 decode 后的字符串取下标是错的
    print(f"\nchosen masked :\n{decode_tokens_from_batch(first_batch["chosen"][0][first_batch["chosen_mask"][0]],tokenizer=tokenizer)}")

    # for batch in train_dataloader:
    #     print(
    #         batch["chosen"].shape,
    #         batch["rejected"].shape,
    #     )
   

if "__main__" == __name__:
    # data = load_data(data_path)
    # dataloader_slice(data)

    test()
    pass