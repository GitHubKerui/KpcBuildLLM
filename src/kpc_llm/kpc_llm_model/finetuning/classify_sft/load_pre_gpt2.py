

from dataclasses import asdict
from llms_from_scratch.ch04 import GPTModel, generate_text_simple
from llms_from_scratch.ch05 import text_to_token_ids, token_ids_to_text
from pathlib import Path
from kpc_llm.kpc_llm_model import KpcLLMModel
from kpc_llm.kpc_llm_model.pretraining.generate_and_print_sample import generate_and_print_sample
from kpc_llm.kpc_llm_model.pretraining.model_save_load import load_model_only
from kpc_llm.kpc_llm_model.token_process.tokenizer_hub import tiktokenizer2idsUnsq, tokenizer2txtsSq
from kpc_llm.kpc_llm_model.train_cfg import GPT2_LARGE_CNF, get_gpt2_cnf
from kpc_llm.kpc_llm_version2.GPTModel_load import to_ch04_config
from kpc_llm.kpc_llm_version2.gpt2_weight_map import load_gpt2_weights_into_kpc
from kpc_llm.utils import  getlogger,get_project_root
from kpc_llm.utils.adaptive_device import get_adaptive_device
from kpc_llm.utils.file_tools import getResourceFromUrl
import tiktoken
import torch
from kpc_llm.kpc_llm_model.token_process.tokenizer_hub import tiktokenizer2idsUnsq,tokenizer2txtsSq

tokenizer = tiktoken.get_encoding("gpt2")
log =  getlogger()

def load_GPT2_model():
    file_name = "gpt2-large-774M.pth"
    model_sv_path = get_project_root()/"checkpoints"/"GPT2"/file_name
    if not model_sv_path.exists():
        model_sv_path = getResourceFromUrl(
            f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
            str(Path("checkpoints") / "GPT2"),
        )
        log.info(f"savePthFilepath : {model_sv_path}")
    
    gpt2_large_cnf = to_ch04_config(GPT2_LARGE_CNF)
    device = get_adaptive_device()
    model = GPTModel(gpt2_large_cnf)
    model = load_model_only(model_sv_path,model)
    model = model.to(device)
    # log.info(f"Model : {model}")
    # log.info(f"Model type: {model.type}")
    return model

def load_GPT2kpc_model():
    file_name = "gpt2-large-774M.pth"
    model_sv_path = get_project_root()/"checkpoints"/"GPT2"/file_name
    cnf = get_gpt2_cnf(file_name)
    if not model_sv_path.exists():
        model_sv_path = getResourceFromUrl(
            f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}",
            str(Path("checkpoints") / "GPT2"),
        )
        log.info(f"savePthFilepath : {model_sv_path}")
    
    model = KpcLLMModel(asdict(cnf))
    device = get_adaptive_device()
    state_dict = torch.load(model_sv_path, map_location=device, weights_only=True)
    model = load_gpt2_weights_into_kpc(model, state_dict)
    model = model.to(device)
    return model

def gpt2_gen_txt(model,cnf,txt_num,prompt="Every effort moves you "):
    # log.info(f"Model type: {model.type}")
    device = get_adaptive_device()
    model.eval()
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=txt_num,
        context_size=cnf["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))
    
# kpc model 有温度 和topk 测试了下 自己的生成更好用。
def gpt2_kpc_gen_txt(model,cnf,txt_num,prompt="Every effort moves you "):
    device = get_adaptive_device()
    model = model.to(device)
    model.eval()  # 关闭所有训练用的 dropout
    torch.manual_seed(825)
    tokenizer = tiktoken.get_encoding("gpt2")
    generate_and_print_sample(model,device,cnf["context_length"],prompt,tiktokenizer2idsUnsq,tokenizer2txtsSq,tokenizer,txt_num,0.8,6)

    

if "__main__" == __name__:

    # gpt2_large_cnf = to_ch04_config(GPT2_LARGE_CNF)
    # model = load_GPT2_model()
    # gpt2_kpc_gen_txt(model,gpt2_large_cnf,100)
    

    pass
