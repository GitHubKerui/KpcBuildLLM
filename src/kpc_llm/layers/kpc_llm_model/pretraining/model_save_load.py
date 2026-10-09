""" 保存模型测试代码 """

from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.train_cfg import GPT2_cl100k_base_CNF
from dataclasses import asdict
from kpc_llm.utils import getlogger
from kpc_llm.utils.adaptive_device import get_adaptive_device
from kpc_llm.utils.prj_dirc_file_tools import create_dir_under_prj,add_flnm_timestamp,get_project_root
import time
import torch

logger =getlogger()

# 只保存模型
def save_model(modelname,model:torch.nn.Module):
    saveDic = "checkpoints"
    # 加上时间戳
    modelname = add_flnm_timestamp(modelname)
    saveFileNm = modelname +".pth"
    # 在项目目录创建保存模型参数的文件夹
    savePath = create_dir_under_prj(saveDic)
    savePath = savePath / saveFileNm
    torch.save(model.state_dict(),savePath)
    logger.info(f"Model successfully saved to: {savePath}")

# 保存模型和优化器
def save_model_optimizer(model:torch.nn.Module,opitmizer:torch.optim.Optimizer,mdl_opt_nm:str= "kpcModelandOptimizer"):
    saveDic = "checkpoints"
    # 加上时间戳
    tmstamp = time.strftime("_%Y%m%d_%H%M%S")
    mdl_opt_nm = mdl_opt_nm + tmstamp
    saveFileNm = mdl_opt_nm +".pth"
    # 在项目目录创建保存模型参数的文件夹
    savePath = create_dir_under_prj(saveDic)
    savePath = savePath / saveFileNm
    torch.save(
    {"model_state_dict":model.state_dict(),"optimizer_state_dict":opitmizer.state_dict()}
    ,savePath
    )
    logger.info(f"Model and optimizer successfully saved to: {savePath}")


# 只加载模型返回模型
def load_model_only(model_path,initial_model:torch.nn.Module):
    device = get_adaptive_device()
    # weights_only=True 时，PyTorch 在底层会直接剥离并禁用 pickle 允许执行任意代码的功能。
    # weights_only=True 必须开这一项，避免执行可能从网上获取的模型初始化阶段里通过pickle嵌入黑客的恶意脚本
    state_dict = torch.load(model_path,map_location=device,weights_only=True)
    initial_model.load_state_dict(state_dict)
    return initial_model

# 加载模型和优化器
def load_model_optimizer(model_optimizer_path,initial_model:torch.nn.Module,initial_optimizer:torch.optim.Optimizer):
    device = get_adaptive_device()
    state_dict = torch.load(model_optimizer_path,map_location=device,weights_only=True)
    model_state_dict =state_dict["model_state_dict"]
    optimizer_state_dict =state_dict["optimizer_state_dict"]
    initial_model.load_state_dict(model_state_dict)
    initial_optimizer.load_state_dict(optimizer_state_dict)
    return initial_model,initial_optimizer

""" 测试代码 """
def save_model_test(modelname):
    cnf = asdict(GPT2_cl100k_base_CNF)
    model = KpcLLMModel(cnf)
    save_model(modelname,model)
    # 这里只是初始化了一个opt，后面记载保存的后，这个配置会被覆盖

def save_model_optimizer_test(model_optimizer_name):
    cnf = asdict(GPT2_cl100k_base_CNF)
    model = KpcLLMModel(cnf)
    # 这里只是初始化了一个opt，后面记载保存的后，这个配置会被覆盖
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    save_model_optimizer(model,optimizer,model_optimizer_name)

def load_model_test(modelfileName):
    cnf = asdict(GPT2_cl100k_base_CNF)
    model = KpcLLMModel(cnf)
    model_path = get_project_root()/"checkpoints"/modelfileName
    model = load_model_only(model_path,model)
    logger.info(f"Model type: {model.type}")

def load_model_optimizer_test(modelOptimfileName):
    cnf = asdict(GPT2_cl100k_base_CNF)
    model = KpcLLMModel(cnf)
    optimizer = torch.optim.AdamW(model.parameters())
    model_optim_path = get_project_root()/"checkpoints"/modelOptimfileName
    model,optimzier = load_model_optimizer(model_optim_path,model,optimizer)
    logger.info(f"model_optim type: {model.type} - {optimizer.__class__}")

# 注：原先本文件末尾的 Top-K 权重保存 + 早停（save_model_topk / TopKModelSaver /
# TopKEntry / _WEIGHT_FILE_RE）已移出，改由 pretraining/topk_saver.py 的 TopKSaver 实现 ——
# 它用 accelerate 的 save_model 存 safetensors（无 pickle 风险、自动解包 DDP），
# 并通过 register_for_checkpointing 让 topk 记录跟着 save_state 一起存档。
# 本文件保留的是与 accelerate 无关的通用 save/load 函数。
