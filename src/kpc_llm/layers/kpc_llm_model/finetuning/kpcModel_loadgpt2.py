"""把官方 GPT-2 权重加载进 KpcLLMModel 并生成一段文本。

用法：只改下面的 file_name（small / medium / large / xl），直接运行即可。
建模配置会按文件名里的规格后缀自动匹配，不需要再手动同步任何字段。
"""
from dataclasses import asdict
from pathlib import Path

import tiktoken
import torch

from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.generate_txt_loop import generate_txt_loop
from kpc_llm.layers.kpc_llm_model.finetuning.gpt2_weight_map import load_gpt2_weights_into_kpc
from kpc_llm.layers.kpc_llm_model.token_process.tokenizer_hub import tiktokenizer2idsUnsq, tokenizer2txtsSq
from kpc_llm.layers.kpc_llm_model.train_cfg import get_gpt2_cnf
from kpc_llm.utils.adaptive_device import get_adaptive_device
from kpc_llm.utils.logger import getlogger
from kpc_llm.utils.prj_dirc_file_tools import getResourceFromUrl

logger = getlogger()

# file_name = "gpt2-small-124M.pth"
# file_name = "gpt2-medium-355M.pth"
file_name = "gpt2-large-774M.pth"
# file_name = "gpt2-xl-1558M.pth"

# 建模规格必须和权重文件一致。原先手写 NEW_CONFIG 时踩过坑：
# 下载了 large(1280/36层) 却被按 small(768/12层) 建模，load_state_dict 全量 mismatch。
# 这里按文件名后缀（small/medium/large/xl）自动匹配 GPT2_*_CNF，从根上避免不一致。
cnf = get_gpt2_cnf(file_name)
logger.info(f"file_name : {file_name} -> {cnf.cnf_name}")
logger.info(f"cnf : {asdict(cnf)}")

savePthFilepath = getResourceFromUrl(
    f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
    str(Path("checkpoints") / "GPT2"),
)
logger.info(f"savePthFilepath : {savePthFilepath}")

# 官方 checkpoint 的 key 命名（tok_emb / trf_blocks / W_query / norm1.scale / out_head ...）
# 与本模型（vocab_emb / trnsf_blocks / W_q / normal1.weight / out_liner ...）不同，
# 必须先经 load_gpt2_weights_into_kpc 做 key 重映射，不能直接 load_state_dict。
device = get_adaptive_device()
model = KpcLLMModel(asdict(cnf))
state_dict = torch.load(savePthFilepath, map_location=device, weights_only=True)
model = load_gpt2_weights_into_kpc(model, state_dict)
model.to(device)
model.eval()  # 关闭所有训练用的 dropout

torch.manual_seed(825)
tokenizer = tiktoken.get_encoding("gpt2")
# 输入必须与模型同设备，否则报 "Expected all tensors to be on the same device"
input_ids = tiktokenizer2idsUnsq("Long long ago, there is a girl ", tokenizer).to(device)

# 必须用采样：generate_txt_loop 的默认 temperature=0.0 / top_k=0 是纯贪心 argmax，
# 开始忘记设置温度和 topk 导致模型重复循环，使用了 argmax 看起来像模型很笨。一旦进入高频词组就会锁死成复读，
prediction_ids = generate_txt_loop(input_ids, cnf.cntext_lnth, model, 150, 0.8, 40)
logger.info(f"prediction_ids : {prediction_ids}")
logger.info(f"prediction_txt : {tokenizer2txtsSq(prediction_ids, tokenizer)}")
