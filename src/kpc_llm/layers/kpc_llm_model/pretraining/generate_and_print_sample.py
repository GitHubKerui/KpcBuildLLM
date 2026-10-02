from kpc_llm.layers.kpc_llm_model.token_process import tokenizer_hub 
from kpc_llm.layers.kpc_llm_model.pretraining.generate_txt_loop import generate_txt_loop
import torch
from kpc_llm.utils.logger import getlogger

logger = getlogger()

# 自回归生成预测测试
def generate_and_print_sample(model, device, context_len,start_context,txt2idsFn,ids2txtFn,tokenizer,generate_len=50):
    model.eval()
    
    # context_size = model.pos_emb.weight.shape[0]
    # input_ids = qwtokenizer2ids(start_context,tokenizer).to(device)
    input_ids = txt2idsFn(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_txt_loop(input_ids,context_len,model,generate_len,1,3)  
        decoded_text = ids2txtFn(token_ids,tokenizer)
        logger.info(decoded_text.replace("\n", " "))  # Compact print format
    # model.train()