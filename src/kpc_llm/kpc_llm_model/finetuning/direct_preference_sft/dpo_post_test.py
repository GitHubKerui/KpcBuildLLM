"""DPO 训练后的效果验证。

从同一份数据里抽 N 条（取 test 切片，训练没见过），对每条打印：
    1. 模型自由生成的内容
    2. 模型给 chosen / rejected 的序列 logprob —— DPO 训练的目标就是这个差值
"""

import tiktoken
import torch
import torch.nn.functional as F

from llms_from_scratch.ch04 import generate_text_simple

from kpc_llm.kpc_llm_model.finetuning.Instruction_sft.load_pre_gpt2_small import load_sft_model
from kpc_llm.kpc_llm_model.finetuning.direct_preference_sft.dpo_dataset import (
    data_path,
    div_data,
    load_data,
    prompt_input,
)
from kpc_llm.utils.adaptive_device import get_adaptive_device

DPO_CKPT = "checkpoints/lightning/dpo/gpt2-medium-355M-dpo-epoch=01-val_loss=0.134.ckpt"
MODEL_FILE = "gpt2-medium-355M.pth"     # 与 checkpoint 规格一致
N_SAMPLES = 5
MAX_NEW_TOKENS = 40


def build_prompt(entry) -> str:
    """必须和数据集构造时完全一致，否则模型看到的分布对不上。"""
    return f"{prompt_input(entry)}\n\n### Response:\n"


@torch.no_grad()
def response_logprob(model, tokenizer, prompt: str, response: str) -> float:
    """response 段的序列 logprob，DPO 判断偏好看的就是这个量。"""
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids + tokenizer.encode(response)],
                       device=next(model.parameters()).device)

    logits = model(ids)                                   # [1, T, V]
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)[0]
    token_logp = logp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    # token_logp[i] 预测的是第 i+1 个 token，response 从索引 len(prompt_ids) 开始
    return token_logp[len(prompt_ids) - 1:].sum().item()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)],
                       device=next(model.parameters()).device)
    out = generate_text_simple(model, ids, max_new_tokens, model.pos_emb.weight.shape[0])
    return tokenizer.decode(out[0].tolist())[len(prompt):].strip()


def test():
    model = load_sft_model(DPO_CKPT, MODEL_FILE).eval()   # DPO ckpt 里的 model.* 就是训练好的 policy
    tokenizer = tiktoken.get_encoding("gpt2")

    data = load_data(data_path)
    _, test_data, _ = div_data(data)
    samples = test_data[:N_SAMPLES]

    hit = 0
    for i, entry in enumerate(samples, 1):
        prompt = build_prompt(entry)
        generated = generate(model, tokenizer, prompt)
        lp_chosen = response_logprob(model, tokenizer, prompt, entry["chosen"])
        lp_rejected = response_logprob(model, tokenizer, prompt, entry["rejected"])
        hit += int(lp_chosen > lp_rejected)

        print(f"\n{'=' * 72}\n[{i}] {entry['instruction']}")
        if entry["input"]:
            print(f"    Input    : {entry['input']}")
        print(f"    Chosen   : {entry['chosen']}")
        print(f"    Rejected : {entry['rejected']}")
        print(f"    模型生成  : {generated[:120]}")
        print(f"    logp(chosen) {lp_chosen:>8.3f}   logp(rejected) {lp_rejected:>8.3f}"
              f"   → 偏向 {'chosen' if lp_chosen > lp_rejected else 'rejected'}")

    print(f"\n偏好命中 {hit}/{len(samples)} = {hit / len(samples):.0%}")


if __name__ == "__main__":
    test()
