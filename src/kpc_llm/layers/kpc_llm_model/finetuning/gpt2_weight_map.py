"""
把 rasbt/gpt2-from-scratch-pytorch 格式的 GPT-2 官方权重，
灌进本项目自己实现的 KpcLLMModel。

为什么需要这个模块：
    两份实现的参数命名完全不同，直接 load_state_dict 会全量
    "Missing key(s)" + "Unexpected key(s)"。这里只做 key 重命名 + 少量补零，
    不改变 torch 的加载语义（仍然 strict=True，漏一个就报错）。

命名对照（checkpoint -> KpcLLMModel）：
    tok_emb.weight                    -> vocab_emb.weight
    pos_emb.weight                    -> pstn_emb.weight
    trf_blocks.{i}.att.W_query.*      -> trnsf_blocks.{i}.attention.W_q.*
    trf_blocks.{i}.att.W_key.*        -> trnsf_blocks.{i}.attention.W_k.*
    trf_blocks.{i}.att.W_value.*      -> trnsf_blocks.{i}.attention.W_v.*
    trf_blocks.{i}.att.out_proj.*     -> trnsf_blocks.{i}.attention.out_proj.*
    trf_blocks.{i}.att.mask           -> (丢弃，本模型用 persistent=False 的 buffer 自行计算)
    trf_blocks.{i}.ff.layers.{0,2}.*  -> trnsf_blocks.{i}.ff.layer.{0,2}.*
    trf_blocks.{i}.norm1.scale/shift  -> trnsf_blocks.{i}.normal1.weight/bias
    trf_blocks.{i}.norm2.scale/shift  -> trnsf_blocks.{i}.normal2.weight/bias
    final_norm.scale/shift            -> final_norml.weight/bias
    out_head.weight                   -> out_liner.weight
    (checkpoint 无)                   -> out_liner.bias，按 0 填充
"""

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module

from kpc_llm.utils.logger import getlogger

logger = getlogger()

# ---------------------------------------------------------------------------
# 转换规则，四类，对应上面命名对照表的全部情形
# ---------------------------------------------------------------------------

# 1) 需要精确匹配的顶层 key（无法用后缀或子串规则表达）
_EXACT_MAP = {
    "tok_emb.weight": "vocab_emb.weight",
    "pos_emb.weight": "pstn_emb.weight",
    "final_norm.scale": "final_norml.weight",
    "final_norm.shift": "final_norml.bias",
    "out_head.weight": "out_liner.weight",
}

# 2) 需要子串替换的 key，按此顺序依次执行
_SUBSTR_RULES = (
    ("trf_blocks.", "trnsf_blocks."),
    (".att.", ".attention."),
    (".W_query.", ".W_q."),
    (".W_key.", ".W_k."),
    (".W_value.", ".W_v."),
    (".ff.layers.", ".ff.layer."),
    (".norm1.scale", ".normal1.weight"),
    (".norm1.shift", ".normal1.bias"),
    (".norm2.scale", ".normal2.weight"),
    (".norm2.shift", ".normal2.bias"),
)

# 3) checkpoint 里有、但本模型不用的 key（因果掩码由模型自己算，不需要存权重）
_IGNORED_SUFFIXES = (".att.mask",)

# 4) 本模型有、checkpoint 没有的 key，按 0 填充。
#    GPT-2 的 out_head 是 bias=False 且与 tok_emb 绑定权重，所以 checkpoint 里没有 bias；
#    本模型 out_liner 带 bias，补 0 等价于无偏置，不影响输出。
_ZERO_FILL_KEYS = ("out_liner.bias",)


def rename_gpt2_key(key: str) -> Optional[str]:
    """
    把单个 checkpoint key 重命名成本模型的 key。

    Returns:
        重命名后的 key；若该权重本模型不需要（如因果掩码），返回 None 表示丢弃。
    """
    if key in _EXACT_MAP:
        return _EXACT_MAP[key]
    if key.endswith(_IGNORED_SUFFIXES):
        return None
    for src, dst in _SUBSTR_RULES:
        key = key.replace(src, dst)
    return key


def convert_gpt2_state_dict(gpt2_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """把整个 checkpoint 的 key 重命名，返回可直接喂给 KpcLLMModel 的 state_dict。"""
    converted: Dict[str, Tensor] = {}
    ignored: list[str] = []
    for key, value in gpt2_state_dict.items():
        new_key = rename_gpt2_key(key)
        if new_key is None:
            ignored.append(key)
        else:
            converted[new_key] = value
    if ignored:
        logger.info(f"忽略 checkpoint 里的 {len(ignored)} 个 key（本模型不使用）: 例如 {ignored[:3]}")
    return converted


def load_gpt2_weights_into_kpc(model: Module, gpt2_state_dict: Dict[str, Tensor]) -> Module:
    """
    把 GPT-2 官方权重加载进 KpcLLMModel，失败时给出可读的原因。

    Args:
        model: 已按对应规格（small/medium/large/xl）实例化好的 KpcLLMModel
        gpt2_state_dict: torch.load(..., weights_only=True) 得到的原始 checkpoint

    Returns:
        加载完成的 model（原地修改并返回）

    Raises:
        KeyError: 转换后仍有本模型需要、但 checkpoint 提供不了的权重（_ZERO_FILL_KEYS 除外）
        RuntimeError: 形状对不上（通常是规格选错），或出现本模型不认识的 key
    """
    model_sd = model.state_dict()
    converted = convert_gpt2_state_dict(gpt2_state_dict)

    # 1) 补齐 checkpoint 没有、但模型需要的权重（目前只有 out_liner.bias）
    filled = [k for k in model_sd if k not in converted and k in _ZERO_FILL_KEYS]
    for key in filled:
        converted[key] = torch.zeros_like(model_sd[key])
    if filled:
        logger.info(f"用 0 填充 checkpoint 未提供的权重: {filled}")

    missing = [k for k in model_sd if k not in converted]
    if missing:
        raise KeyError(
            f"checkpoint 缺少本模型需要的权重: {missing[:10]}。"
            f"这通常意味着 model 的规格和 checkpoint 不匹配，或两者结构实现有差异。"
        )

    # 2) 多余的 key：本模型完全不认识，通常说明规格选错了
    extra = [k for k in converted if k not in model_sd]
    if extra:
        raise RuntimeError(f"checkpoint 中存在本模型不认识的 key: {extra[:10]}")

    # 3) 形状预检：先给出可读的错误，避免 PyTorch 抛出一大串 size mismatch
    mismatched = [
        f"  {k}: checkpoint {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
        for k, v in converted.items()
        if tuple(v.shape) != tuple(model_sd[k].shape)
    ]
    if mismatched:
        raise RuntimeError(
            "权重形状不匹配，请确认 model 的规格与 checkpoint 一致：\n" + "\n".join(mismatched[:10])
        )

    # 4) strict=True：任何遗漏都直接报错，避免"静默少加载一层"这种难查的问题
    model.load_state_dict(converted, strict=True)
    logger.info(f"已加载 {len(converted)} 个权重到 {model.__class__.__name__}")
    return model
