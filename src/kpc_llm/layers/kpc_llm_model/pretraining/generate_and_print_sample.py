from _pytest.tmpdir import tmp_path

from kpc_llm.layers.kpc_llm_model.token_process import tokenizer_hub 
from kpc_llm.layers.kpc_llm_model.pretraining.generate_txt_loop import generate_txt_loop
import torch
from kpc_llm.utils.logger import getlogger
import tiktoken
from dataclasses import asdict
from accelerate import Accelerator
from safetensors.torch import load_model as load_safetensors
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.topk_saver import TopKSaver
from kpc_llm.layers.kpc_llm_model.train_cfg import GPT2_cl100k_base_CNF
from kpc_llm.utils.adaptive_device import get_adaptive_device
from kpc_llm.utils.prj_dirc_file_tools import get_project_root

logger = getlogger()

# 自回归生成预测测试
def generate_and_print_sample(model, device, context_len,start_context,txt2idsFn,ids2txtFn,tokenizer,generate_len=50,toptmp:float=1.0,topk:int=3):
    model.eval()
    
    # context_size = model.pos_emb.weight.shape[0]
    # input_ids = qwtokenizer2ids(start_context,tokenizer).to(device)
    input_ids = txt2idsFn(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_txt_loop(input_ids,context_len,model,generate_len,toptmp,topk)  
        decoded_text = ids2txtFn(token_ids,tokenizer)
        logger.info(decoded_text.replace("\n", " "))  # Compact print format
    # model.train()


def generate_from_best_checkpoint(
    cnf=asdict(GPT2_cl100k_base_CNF),
    topk_root_dir=str(get_project_root() / "checkpoints" / "topk_safetensors"),
    start_context="Long long ago, there is a girl ",
    generate_len=250,
    toptmp:float=1.0,
    topk=3
):
    """加载 checkpoints/topk_safetensors/kpcModel 最优模型并回归生成内容。

    复用 TopKSaver.best_dir()：它按 topk_meta.json 的排名取 entries[0]，即历史最优那份。

    Args:
        cnf:           模型配置 dict（默认 GPT2_cl100k_base_CNF）。
        topk_root_dir: topk 权重根目录。
        start_context: 生成用的起始文本，应与训练时一致。
        generate_len:  生成的 token 数。
    """

    # 1) 定位历史最优权重目录。TopKSaver 构造时会 register_for_checkpointing，
    #    推理场景只需一个空的 accelerator 满足接口即可。
    saver = TopKSaver(Accelerator(), topk=1, root_dir=topk_root_dir)
    best_dir = saver.best_dir()
    if best_dir is None:
        logger.error(f"{saver.root_dir} 下没有可用权重，请先训练")
        raise SystemExit(1)
    best_file = best_dir / "model.safetensors"
    logger.info(f"加载最优权重：{best_file}（val_loss={saver.best_val_loss:.4f}）")

    # 2) 构造与训练同配置的模型并加载权重。保存端已经 unwrap_model，
    #    safetensors 里的键与裸 KpcLLMModel 完全一致，可直接 load。
    device = get_adaptive_device()
    model = KpcLLMModel(cnf)
    load_safetensors(model, str(best_file))
    model.to(device)

    # 3) 回归生成
    tokenizer = tiktoken.get_encoding("cl100k_base")
    generate_and_print_sample(
        model, device, cnf["cntext_lnth"], start_context,
        tokenizer_hub.tiktokenizer2idsUnsq, tokenizer_hub.tokenizer2txtsSq,
        tokenizer, generate_len,toptmp,topk)


if __name__ == "__main__":
    cnf=asdict(GPT2_cl100k_base_CNF)
    topk_root_dir=str(get_project_root() / "checkpoints" / "topk_safetensors")
    start_context="Long long ago, there is a girl "
    generate_len=250
    tmp=0.7
    topk=5

    generate_from_best_checkpoint(cnf,topk_root_dir,start_context,generate_len,tmp,topk)


