"""
显存需求计算模块 / GPU Memory Requirement Calculator
====================================================

【这个模块解决什么问题 / What it solves】
训练前先算清楚"这个配置到底要吃多少显存"，避免跑到一半 OOM（Out Of Memory）崩掉。

【处理流程 / Processing flow】一共 5 步：

    STEP 1  统计参数量      count_parameters()
            └── 按组件拆开算：词嵌入 / 位置嵌入 / N 个 TransformerBlock / 最终 LayerNorm / 输出头

    STEP 2  常驻显存        calc_resident_memory()
            └── 参数(weights) + 梯度(grads) + 优化器状态(optimizer states)
                这部分"与 batch_size 无关"，模型一建好就固定占着

    STEP 3  激活显存        calc_activation_memory()
            └── 前向传播的中间结果，**与 batch_size 成正比**
                其中"输出 logits"是绝对大头（词表 151936 时占 90% 以上）

    STEP 4  汇总对比        report()
            └── 常驻 + 激活 = 峰值显存，再和你的真实 GPU 显存比一比，给出结论

    STEP 5  反推 batch      suggest_max_batch_size()
            └── 已知显存上限，反推"最大能开多大的 batch_size"

【显存构成的直觉 / Key intuition】
    ┌─ 常驻部分（固定）─────────────────────────────┐
    │  参数 1 份  +  梯度 1 份  +  AdamW 状态 2 份    │  → fp32 下 = 参数量 × 16 字节
    └───────────────────────────────────────────────┘
    ┌─ 激活部分（随 batch 线性增长）─────────────────┐
    │  输出 logits × 3 份（前向 / log_softmax / 梯度）│  → B × T × 词表 × 4字节 × 3
    │  + 每层隐藏激活（经验估算）                     │
    └───────────────────────────────────────────────┘

【用法 / Usage】
    python hardware_require_calcu.py
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 常量区 / Constants
# ---------------------------------------------------------------------------

# 各精度占用的字节数 / Bytes per element for each dtype
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}

# 每种优化器"每个参数"额外要存几份状态 / Extra states per parameter
#   SGD          : 0 份（只存参数 + 梯度）
#   SGD+momentum : 1 份（momentum）
#   Adam/AdamW   : 2 份（一阶动量 exp_avg + 二阶动量 exp_avg_sq）
OPTIMIZER_STATES = {
    "sgd": 0,
    "sgd_momentum": 1,
    "rmsprop": 1,
    "adam": 2,
    "adamw": 2,
}

GB = 1024 ** 3  # 1 GB = 1024^3 字节（注意：厂商标称的 8 GB 通常也是这个量级）


# ---------------------------------------------------------------------------
# STEP 1：统计参数量 / Count parameters
# ---------------------------------------------------------------------------
def count_parameters(cnf: dict) -> dict:
    """按组件统计模型的参数量。

    Args:
        cnf: 模型配置字典

    Returns:
        dict，key 为组件名，value 为参数个数（不是字节）
    """
    vocab_size = cnf["vcab_sz"]                  # 词表大小 V
    emb_dim = cnf["emb_dim"]                     # 嵌入维度 E
    block_num = cnf["trnsf_blocks_num"]          # TransformerBlock 层数 L
    pos_len = cnf["max_cntxt_pstion_lnth"]       # 位置编码表长度 T_max
    use_bias = cnf.get("qkv_bias", False)        # 线性层是否带 bias

    # 1) token 嵌入表：[V, E]
    vocab_emb = vocab_size * emb_dim

    # 2) 位置嵌入表：[T_max, E]
    pos_emb = pos_len * emb_dim

    # 3) 单个 TransformerBlock 的参数量
    #    3.1 注意力：W_q / W_k / W_v，每个都是 [E, E]（+ bias 则 +E）
    attn_qkv = 3 * (emb_dim * emb_dim + (emb_dim if use_bias else 0))
    #    3.2 FFN：升维 E→4E（+4E bias），再降维 4E→E（+E bias）
    ffn_up = emb_dim * 4 * emb_dim + 4 * emb_dim
    ffn_down = 4 * emb_dim * emb_dim + emb_dim
    per_block = attn_qkv + ffn_up + ffn_down

    #    3.3 乘上层数
    blocks = per_block * block_num

    # 4) 最终的 LayerNorm：可学习的 weight + bias，各 E 个
    final_norm = 2 * emb_dim

    # 5) 输出头 out_liner：[E, V]（bias 由 qkv_bias 决定）
    out_head = emb_dim * vocab_size + (vocab_size if use_bias else 0)

    total = vocab_emb + pos_emb + blocks + final_norm + out_head

    return {
        "vocab_emb": vocab_emb,
        "pos_emb": pos_emb,
        "per_block": per_block,
        "blocks": blocks,
        "final_norm": final_norm,
        "out_head": out_head,
        "total": total,
    }


# ---------------------------------------------------------------------------
# STEP 2：常驻显存 / Resident memory (params + grads + optimizer)
# ---------------------------------------------------------------------------
def calc_resident_memory(cnf: dict, param_bytes: int = 4, optimizer: str = "adamw") -> dict:
    """计算与 batch_size 无关的"常驻显存"。

    组成：
        参数 weights        : 1 份
        梯度 grads          : 1 份（反向传播时每个参数都要存梯度）
        优化器状态 states   : AdamW = 2 份

    Args:
        cnf: 模型配置
        param_bytes: 参数/梯度/优化器状态的字节数。fp32 = 4（PyTorch 默认）
        optimizer: 优化器类型，决定状态份数

    Returns:
        dict，value 单位为字节
    """
    params = count_parameters(cnf)
    total_params = params["total"]

    state_num = OPTIMIZER_STATES.get(optimizer.lower(), 2)

    weights = total_params * param_bytes
    grads = total_params * param_bytes
    states = total_params * param_bytes * state_num

    return {
        "weights": weights,
        "grads": grads,
        "optimizer_states": states,
        "total": weights + grads + states,
        "state_num": state_num,
    }


# ---------------------------------------------------------------------------
# STEP 3：激活显存 / Activation memory (depends on batch_size)
# ---------------------------------------------------------------------------
def calc_activation_memory(
    cnf: dict,
    batch_size: int,
    chunk_len: int,
    act_bytes: int = 4,
) -> dict:
    """计算前向传播的中间激活显存，与 batch_size 成正比。

    两大块：
        A. 输出 logits（绝对大头）
           cross_entropy 在实际计算中会产生 3 份同尺寸张量：
             1) 模型输出的 logits            [B, T, V]
             2) cross_entropy 内部 log_softmax 的结果（要留给反向传播）
             3) 反向传播时 logits 的梯度
           → 3 × B × T × V × act_bytes

        B. 每层隐藏激活（经验估算，量级远小于 A）
           每个 token 每层约需要：
             ~20 × E 个元素（LayerNorm 输入、Q/K/V、残差、FFN 的 4E 升维与激活值……）
             + 2 × heads_num × T 个元素（注意力分数矩阵 + softmax 结果，形状 [H, T, T]）

    Args:
        cnf: 模型配置
        batch_size: 批大小 B
        chunk_len: 上下文/序列长度 T
        act_bytes: 激活值字节数。fp32 = 4，bf16/fp16 = 2

    Returns:
        dict，value 单位为字节
    """
    vocab_size = cnf["vcab_sz"]
    emb_dim = cnf["emb_dim"]
    block_num = cnf["trnsf_blocks_num"]
    heads_num = cnf.get("heads_num", 1)

    # A. 输出 logits（3 份）
    logits_bytes = 3 * batch_size * chunk_len * vocab_size * act_bytes

    # B. 隐藏激活（经验系数，用于量级估算）
    elem_per_token_per_layer = 20 * emb_dim + 2 * heads_num * chunk_len
    hidden_bytes = block_num * batch_size * chunk_len * elem_per_token_per_layer * act_bytes

    return {
        "logits": logits_bytes,
        "hidden": hidden_bytes,
        "total": logits_bytes + hidden_bytes,
    }


# ---------------------------------------------------------------------------
# STEP 4：汇总 / Total peak memory
# ---------------------------------------------------------------------------
def calc_total_memory(
    cnf: dict,
    batch_size: int | None = None,
    chunk_len: int | None = None,
    param_bytes: int = 4,
    act_bytes: int = 4,
    optimizer: str = "adamw",
) -> dict:
    """汇总训练时的峰值显存。

    Args:
        cnf: 模型配置
        batch_size: 批大小，缺省则取 cnf["batch_size"]，再缺省用 4
        chunk_len: 序列长度，缺省取 cnf["cntext_lnth"]
        param_bytes: 参数精度字节数（权重/梯度/优化器状态）
        act_bytes: 激活值精度字节数（bf16 混合精度时填 2）
        optimizer: 优化器类型

    Returns:
        dict，包含各项明细（字节）与 total_gb（GB）
    """
    if batch_size is None:
        batch_size = cnf.get("batch_size", 4)
    if chunk_len is None:
        chunk_len = cnf.get("cntext_lnth", 256)

    resident = calc_resident_memory(cnf, param_bytes, optimizer)
    activation = calc_activation_memory(cnf, batch_size, chunk_len, act_bytes)

    total = resident["total"] + activation["total"]

    return {
        "batch_size": batch_size,
        "chunk_len": chunk_len,
        "resident": resident,
        "activation": activation,
        "total": total,
        "total_gb": total / GB,
    }


# ---------------------------------------------------------------------------
# STEP 5：反推最大 batch_size / Suggest max batch size
# ---------------------------------------------------------------------------
def suggest_max_batch_size(
    cnf: dict,
    vram_gb: float,
    chunk_len: int | None = None,
    param_bytes: int = 4,
    act_bytes: int = 4,
    optimizer: str = "adamw",
    safety_ratio: float = 0.9,
) -> int:
    """在给定显存上限下，反推最大可用 batch_size。

    Args:
        vram_gb: 显卡总显存（GB）
        safety_ratio: 安全系数，只使用显存的这个比例（默认 0.9，给碎片和临时缓冲留余量）

    Returns:
        最大 batch_size（至少为 1）
    """
    if chunk_len is None:
        chunk_len = cnf.get("cntext_lnth", 256)

    budget = vram_gb * GB * safety_ratio
    resident_bytes = calc_resident_memory(cnf, param_bytes, optimizer)["total"]
    usable = budget - resident_bytes

    if usable <= 0:
        return 1  # 常驻显存就已经超了，只能换更小的模型/精度

    # 单个 batch 样本需要的激活显存
    per_sample = calc_activation_memory(cnf, 1, chunk_len, act_bytes)["total"]
    max_bs = int(usable // per_sample) if per_sample > 0 else 1

    return max(max_bs, 1)


# ---------------------------------------------------------------------------
# 显卡信息 / GPU info
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict | None:
    """读取当前 GPU 的名称与总显存；没有可用 CUDA 时返回 None。"""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_gb": props.total_memory / GB,
    }


# ---------------------------------------------------------------------------
# 人类可读的报告 / Human-readable report
# ---------------------------------------------------------------------------
def report(
    cnf: dict,
    batch_size: int | None = None,
    param_bytes: int = 4,
    act_bytes: int = 4,
    optimizer: str = "adamw",
) -> None:
    """打印一份完整的显存报告，并和本机真实 GPU 做对比。"""
    params = count_parameters(cnf)
    result = calc_total_memory(cnf, batch_size, None, param_bytes, act_bytes, optimizer)

    line = "=" * 62
    print(line)
    print("显存需求报告 / GPU Memory Report")
    print(line)

    # STEP 1 参数量
    print("\n[STEP 1] 参数量 / Parameters")
    print(f"  词嵌入 vocab_emb      : {params['vocab_emb'] / 1e6:>10.2f} M")
    print(f"  位置嵌入 pos_emb      : {params['pos_emb'] / 1e6:>10.2f} M")
    print(f"  Transformer blocks    : {params['blocks'] / 1e6:>10.2f} M"
          f"  (每层 {params['per_block'] / 1e6:.2f} M × {cnf['trnsf_blocks_num']} 层)")
    print(f"  最终 LayerNorm        : {params['final_norm'] / 1e6:>10.2f} M")
    print(f"  输出头 out_head       : {params['out_head'] / 1e6:>10.2f} M")
    print(f"  {'-' * 56}")
    print(f"  合计 total            : {params['total'] / 1e6:>10.2f} M")

    # STEP 2 常驻显存
    r = result["resident"]
    print(f"\n[STEP 2] 常驻显存 / Resident ({optimizer}，fp32)")
    print(f"  参数 weights          : {r['weights'] / GB:>8.2f} GB")
    print(f"  梯度 grads            : {r['grads'] / GB:>8.2f} GB")
    print(f"  优化器状态 states ×{r['state_num']}  : {r['optimizer_states'] / GB:>8.2f} GB")
    print(f"  小计                  : {r['total'] / GB:>8.2f} GB   ← 与 batch_size 无关")

    # STEP 3 激活显存
    a = result["activation"]
    print(f"\n[STEP 3] 激活显存 / Activation (batch_size={result['batch_size']}, "
          f"chunk_len={result['chunk_len']})")
    print(f"  输出 logits ×3        : {a['logits'] / GB:>8.2f} GB   ← 绝对大头")
    print(f"  隐藏激活(经验估算)    : {a['hidden'] / GB:>8.2f} GB")
    print(f"  小计                  : {a['total'] / GB:>8.2f} GB   ← 随 batch_size 线性增长")

    # STEP 4 汇总
    print(f"\n[STEP 4] 峰值显存 / Peak")
    print(f"  预计占用              : {result['total_gb']:>8.2f} GB")

    gpu = get_gpu_info()
    if gpu:
        used_ratio = result["total_gb"] / gpu["total_gb"]
        print(f"  本机显卡              : {gpu['name']}")
        print(f"  显卡总显存            : {gpu['total_gb']:>8.2f} GB")
        print(f"  占用比例              : {used_ratio:>8.1%}")
        if used_ratio > 1.0:
            print("  结论                  : ❌ 会 OOM，请降低 batch_size 或改用 bf16")
        elif used_ratio > 0.85:
            print("  结论                  : ⚠️  非常危险，随时可能 OOM，建议留出更多余量")
        else:
            print("  结论                  : ✅ 可以跑")

        # STEP 5 反推
        max_bs = suggest_max_batch_size(
            cnf, gpu["total_gb"], result["chunk_len"], param_bytes, act_bytes, optimizer)
        print(f"\n[STEP 5] 安全阈值内的最大 batch_size : {max_bs}")

        # 额外给一个 bf16 混合精度的参考
        if act_bytes == 4:
            bf16 = calc_total_memory(
                cnf, result["batch_size"], result["chunk_len"], 4, 2, optimizer)
            bf16_max = suggest_max_batch_size(
                cnf, gpu["total_gb"], result["chunk_len"], 4, 2, optimizer)
            print(f"\n[参考] 开启 bf16 混合精度后 / With bf16 AMP")
            print(f"  同 batch_size 占用    : {bf16['total_gb']:>8.2f} GB")
            print(f"  最大 batch_size       : {bf16_max:>8}")
    else:
        print("  本机显卡              : 未检测到可用 CUDA，跳过对比")

    print("\n" + line)


# ---------------------------------------------------------------------------
# 示例配置 / Demo config
# ---------------------------------------------------------------------------
TRAIN_CNF = {
    "vcab_sz": 151936,                  # 词表大小（Qwen2.5 词表 151665，这里取对齐后的 151936）
    "cntext_lnth": 256,                 # 训练时的上下文长度
    # 最大上下文位置长度，用于生成上下文位置 id，必须 >= 上面的 cntext_lnth
    "max_cntxt_pstion_lnth": 256,
    "emb_dim": 512,                     # 嵌入维度
    "heads_num": 8,                     # 注意力头数（要能整除 emb_dim）
    "trnsf_blocks_num": 8,              # TransformerBlock 层数
    "drop_rt": 0.1,                     # dropout 比例
    "qkv_bias": False,                  # 线性层是否带 bias
    "batch_size": 4,                    # 批大小
    # 是返回 softmax 还是返回 logits 的 argmax 最大值 index，也就是 tokenid
    "returnSoftmax": False,
}


if __name__ == "__main__":
    report(TRAIN_CNF)
