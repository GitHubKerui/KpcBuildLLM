# Description: 测试kpc llm生成txt
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.token_process.tiktokenizer import tiktokenizer2ids, tiktokenizer2txts
from kpc_llm.layers.kpc_llm_model.test.generate_txt_loop import generate_txt_loop
from kpc_llm.utils.logger import getlogger
from torch import manual_seed

logger = getlogger()


# 写配置
KPC_LLM_CONFIG_TEST = {
    "vcab_sz" : 50524,
    "cntext_lnth" : 1024,
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

input_ids = tiktokenizer2ids(input_txt)
# logger.info(f'input_ids : {input_ids}')

prediction_ids = generate_txt_loop(input_ids,8,kpc_llm_model,25)
logger.info(f'prediction_ids : {prediction_ids}')
logger.info(f'prediction_txt : {tiktokenizer2txts(prediction_ids)}')