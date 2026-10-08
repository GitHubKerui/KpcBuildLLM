"""加载官方 GPT-2 权重并生成文本 —— llms_from_scratch（原书）实现版本。

与 kpcModel_loadgpt2.py 的区别：
    本文件用原书的 GPTModel 建模，它的参数命名与官方 checkpoint 完全一致，
    所以能直接 load_state_dict，不需要 key 重映射。
    kpcModel_loadgpt2.py 则把权重灌进本项目的 KpcLLMModel，需要过一层
    gpt2_weight_map 做重命名。
    两个脚本跑同样的 prompt，可以用来对照"原书实现"与"本项目实现"行为是否一致。

用法：只改下面的 file_name，直接运行。
"""
from pathlib import Path

import tiktoken
import torch
from llms_from_scratch.ch04 import GPTModel

from kpc_llm.layers.kpc_llm_model.pretraining.generate_txt_loop import generate_txt_loop
from kpc_llm.layers.kpc_llm_model.token_process.tokenizer_hub import tiktokenizer2idsUnsq, tokenizer2txtsSq
from kpc_llm.layers.kpc_llm_model.train_cfg import TrainConfig, get_gpt2_cnf
from kpc_llm.utils.adaptive_device import get_adaptive_device
from kpc_llm.utils.logger import getlogger
from kpc_llm.utils.prj_dirc_file_tools import getResourceFromUrl

logger = getlogger()


def to_ch04_config(cnf: TrainConfig) -> dict:
    """
    把本项目的 TrainConfig 转成 llms_from_scratch.ch04.GPTModel 认识的字典。

    两边只是字段命名不同，取值一一对应：
        vcab_sz -> vocab_size              heads_num         -> n_heads
        emb_dim -> emb_dim                 trnsf_blocks_num  -> n_layers
        max_cntxt_pstion_lnth -> context_length
        drop_rt -> drop_rate               qkv_bias -> qkv_bias

    这样就不必再单独维护一份 GPT-2 规格表：改了 file_name，
    这里和 train_cfg 里的 GPT2_*_CNF 永远同步，不会再出现
    "下载了 large 却按 small 建模、state_dict 全量 mismatch" 那类不一致。
    """
    return {
        "vocab_size": cnf.vcab_sz,
        "context_length": cnf.max_cntxt_pstion_lnth,
        "emb_dim": cnf.emb_dim,
        "n_heads": cnf.heads_num,
        "n_layers": cnf.trnsf_blocks_num,
        "drop_rate": cnf.drop_rt,
        "qkv_bias": cnf.qkv_bias,
    }


# file_name = "gpt2-small-124M.pth"
# file_name = "gpt2-medium-355M.pth"
file_name = "gpt2-large-774M.pth"
# file_name = "gpt2-xl-1558M.pth"

# 按文件名后缀自动匹配规格，省掉手写规格表的出错可能
cnf = get_gpt2_cnf(file_name)
ch04_config = to_ch04_config(cnf)
logger.info(f"file_name : {file_name} -> {cnf.cnf_name}")
logger.info(f"ch04_config : {ch04_config}")

savePthFilepath = getResourceFromUrl(
    f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
    str(Path("checkpoints") / "GPT2"),
)
logger.info(f"savePthFilepath : {savePthFilepath}")

device = get_adaptive_device()
model = GPTModel(ch04_config)
# 原书实现的参数命名与 checkpoint 一致，可以直接加载
state_dict = torch.load(savePthFilepath, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # 关闭所有训练用的 dropout，因此 drop_rate 取 0.1 还是 0.0 对推理结果无影响

torch.manual_seed(825)
tokenizer = tiktoken.get_encoding("gpt2")
# 输入必须与模型同设备，否则报 "Expected all tensors to be on the same device"
input_ids = tiktokenizer2idsUnsq("Long long ago, there is a girl ", tokenizer).to(device)

# 必须用采样：generate_txt_loop 的默认 temperature=0.0 / top_k=0 是纯贪心 argmax，
# 一旦进入高频词组就会锁死成复读，看起来像模型很笨，实际是解码策略的问题。
prediction_ids = generate_txt_loop(input_ids, ch04_config["context_length"], model, 150, 0.8, 40)
logger.info(f"prediction_ids : {prediction_ids}")
logger.info(f"prediction_txt : {tokenizer2txtsSq(prediction_ids, tokenizer)}")
