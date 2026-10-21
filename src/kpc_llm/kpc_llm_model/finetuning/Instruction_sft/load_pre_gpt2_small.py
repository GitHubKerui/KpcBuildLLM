"""加载 GPT-2 small 124M 预训练权重（llms_from_scratch 原书 GPTModel）。

为什么不复用 classify_sft/load_pre_gpt2.py：
    那个文件里的 load_GPT2_model() 写死了 "gpt2-large-774M.pth" 与 GPT2_LARGE_CNF，
    建出来的 GPTModel 是 1280 维 / 36 层，与本任务要求的 small(768/12) 结构不匹配，
    load_state_dict 会全量 mismatch。此处按文件名自动取规格，避免手抄规格表出错。
"""

from pathlib import Path

import torch
from llms_from_scratch.ch04 import GPTModel

from kpc_llm.kpc_llm_model.train_cfg import get_gpt2_cnf
from kpc_llm.kpc_llm_version2.GPTModel_load import to_ch04_config
from kpc_llm.utils import get_project_root, getlogger
from kpc_llm.utils.adaptive_device import get_adaptive_device
from kpc_llm.utils.file_tools import getResourceFromUrl

log = getlogger()

GPT2_SMALL_FILE = "gpt2-small-124M.pth"
GPT2_MEDIUM_FILE = "gpt2-medium-355M.pth"
GPT2_LARGE_FILE = "gpt2-large-774M.pth"


def load_gpt2_small_model(file_name: str = GPT2_SMALL_FILE) -> GPTModel:
    """按文件名后缀自动匹配 GPT-2 规格并加载预训练权重，返回已 on device 的模型。"""
    # "gpt2-small-124M.pth" -> 命中 "small" -> GPT2_SMALL_CNF(768/12/12/ctx1024)
    cnf = get_gpt2_cnf(file_name)
    ch04_config = to_ch04_config(cnf)
    log.info(f"加载 {file_name} -> {cnf.cnf_name} | {ch04_config}")

    save_path = get_project_root() / "checkpoints" / "GPT2" / file_name
    if not save_path.exists():
        save_path = getResourceFromUrl(
            f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
            str(Path("checkpoints") / "GPT2"),
        )
        log.info(f"下载权重到: {save_path}")

    device = get_adaptive_device()
    model = GPTModel(ch04_config)
    # 原书 GPTModel 的参数命名与官方 checkpoint 一致，可直接 load_state_dict。
    # weights_only=True：只反序列化张量，避免 pickle 执行任意代码。
    state_dict = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)

def load_gpt2_large_model(file_name: str = GPT2_LARGE_FILE) -> GPTModel:
    """按文件名后缀自动匹配 GPT-2 规格并加载预训练权重，返回已 on device 的模型。"""
    cnf = get_gpt2_cnf(file_name)
    ch04_config = to_ch04_config(cnf)
    log.info(f"加载 {file_name} -> {cnf.cnf_name} | {ch04_config}")

    save_path = get_project_root() / "checkpoints" / "GPT2" / file_name
    if not save_path.exists():
        save_path = getResourceFromUrl(
            f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
            str(Path("checkpoints") / "GPT2"),
        )
        log.info(f"下载权重到: {save_path}")

    device = get_adaptive_device()
    model = GPTModel(ch04_config)
    # 原书 GPTModel 的参数命名与官方 checkpoint 一致，可直接 load_state_dict。
    # weights_only=True：只反序列化张量，避免 pickle 执行任意代码。
    state_dict = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model

def load_gpt2_model(file_name: str = GPT2_MEDIUM_FILE) -> GPTModel:
    """按文件名后缀自动匹配 GPT-2 规格并加载预训练权重，返回已 on device 的模型。"""
    cnf = get_gpt2_cnf(file_name)
    ch04_config = to_ch04_config(cnf)
    log.info(f"加载 {file_name} -> {cnf.cnf_name} | {ch04_config}")

    save_path = get_project_root() / "checkpoints" / "GPT2" / file_name
    if not save_path.exists():
        save_path = getResourceFromUrl(
            f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
            str(Path("checkpoints") / "GPT2"),
        )
        log.info(f"下载权重到: {save_path}")

    device = get_adaptive_device()
    model = GPTModel(ch04_config)
    # 原书 GPTModel 的参数命名与官方 checkpoint 一致，可直接 load_state_dict。
    # weights_only=True：只反序列化张量，避免 pickle 执行任意代码。
    state_dict = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model

def load_sft_model(ckpt_path: str, file_name: str = GPT2_SMALL_FILE) -> GPTModel:
    """加载 Lightning 训练出的 SFT checkpoint（DPO 的 policy / reference 起点）。

    Lightning 的 save_weights_only=True 存的是 LightingModule.state_dict()，
    参数名带 "model." 前缀（因为模块里是 self.model = GPTModel(...)），这里剥掉再加载。
    """
    cnf = get_gpt2_cnf(file_name)
    ch04_config = to_ch04_config(cnf)
    model = GPTModel(ch04_config)

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():                      # 允许传项目相对路径
        ckpt_file = get_project_root() / ckpt_file

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    prefix = "model."
    sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    # strict=True：多一个或少一个参数都要报错，避免静默加载了半个模型
    model.load_state_dict(sd, strict=True)

    log.info(f"加载 SFT checkpoint: {ckpt_file} | {len(sd)} 个参数 | {cnf.cnf_name}")
    return model.to(get_adaptive_device())


def test():
    device = get_adaptive_device()
    m = load_gpt2_small_model().to(device)
    n = sum(p.numel() for p in m.parameters())
    print(f"参数量 {n/1e6:.1f}M (应 ≈124M)")
    print("logits shape:", m(torch.randint(0, 50257, (1, 8)).to(device)).shape)  # 应为 [1, 8, 50257]
    # 测试通过

if "__main__" == __name__:
    test()