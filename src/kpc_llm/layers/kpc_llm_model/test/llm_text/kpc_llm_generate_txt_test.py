# Description: 测试kpc llm生成txt
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.token_process.tokenizer_hub import qwtokenizer2ids,tiktokenizer2txts
from kpc_llm.layers.kpc_llm_model.test.llm_text.generate_txt_loop import generate_txt_loop
from kpc_llm.utils.logger import getlogger
from transformers import AutoTokenizer
from torch import manual_seed

logger = getlogger()


# 写配置
KPC_LLM_CONFIG_TEST = {
    "vcab_sz" : 50524,
    "cntext_lnth" : 8,
    # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
    "max_cntxt_pstion_lnth" : 8,
    "emb_dim" : 768,
    "heads_num" : 8,
    "trnsf_blocks_num" : 8,
    "drop_rt" : 0.3,
    "qkv_bias" : False,
    # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
    "returnSoftmax" : False
}
manual_seed(825)
# 实例化LLM
kpc_llm_model = KpcLLMModel(KPC_LLM_CONFIG_TEST)
# 关闭所有的训练用 dropout
kpc_llm_model.eval()

input_txt="""The morning sunlight is filtering through the leaves, 
casting dancing shadows on the ground.In thesunlit terraces of someunknownPlace."""

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
input_ids = qwtokenizer2ids(input_txt,tokenizer)
# logger.info(f'input_ids : {input_ids}')

prediction_ids = generate_txt_loop(input_ids,KPC_LLM_CONFIG_TEST['cntext_lnth'],kpc_llm_model,25)
logger.info(f'prediction_ids : {prediction_ids}')
logger.info(f'prediction_txt : {tiktokenizer2txts(prediction_ids,tokenizer)}')