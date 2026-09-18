

from torch import nn,Tensor,manual_seed,arange,argmax,softmax
from kpc_llm.utils.logger import getlogger
from kpc_llm.layers.kpc_llm_model.kpc_llm_config import KPC_GPT_CONFIG_124M as cnf_kpc
from kpc_llm.layers.kpc_llm_model.transformer_block.transformer_block import KpcTransformerBlock
from kpc_llm.layers.kpc_llm_model.kpc_norml import KpcNormal


KPC_LLM_CONFIG_124M = {
    "vcab_sz" : 50524,
    "cntext_lnth" : 1024,
    "emb_dim" : 768,
    "heads_num" : 12,
    "trnsf_blocks_num" : 12,
    "drop_rt" : 0.1,
    "qkv_bias" : False,
    # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
    "returnSoftmax" : False
}

class KpcLLMModel(nn.Module):
    def __init__(self,cnf) -> None:
        super().__init__()
        vcab_sz = cnf['vcab_sz']
        emb_dim = cnf['emb_dim']
        cntext_lnth = cnf['cntext_lnth']
        # transformer_block的层数
        trnsf_blocks_num = cnf['trnsf_blocks_num']
        # 是否是有偏的布尔值
        qkv_bias = cnf['qkv_bias']
        drop_rt = cnf['drop_rt']
        heads_num = cnf['heads_num']

        self.vocab_emb = nn.Embedding(vcab_sz,emb_dim)
        self.pstn_emb = nn.Embedding( cntext_lnth,emb_dim)
        self.drop = nn.Dropout(drop_rt)
        self.trnsf_blocks = nn.Sequential(*[KpcTransformerBlock(cnf) for _ in range(trnsf_blocks_num)])
        self.final_norml = KpcNormal(cnf['emb_dim'])
        self.out_liner = nn.Linear(cnf['emb_dim'],cnf['vcab_sz'],bias=qkv_bias)

    def forward(self,input):
        btch,cntext_lnth =  input.shape
        # 模型前端token embedding 处理部分
        input_tokens = self.vocab_emb(input)
        input_tokens = input_tokens + self.pstn_emb(arange(cntext_lnth,device=input.device))
        x = self.drop(input_tokens)

        # 模型transformer block 处理部分
        x = self.trnsf_blocks(x)

        # 模型Feed Forword 处理部分

        # 模型最后段 线性 linear层处理部分
        x = self.final_norml(x)
        # 最终返回未经 softmax 处理的 logits
        logits = self.out_liner(x)
        # 如果需要可以增加 softmax的处理.
        if self.returnSoftmax:
            return softmax(logits,dim=-1)
        else:
            return argmax(logits,dim=-1,keepdim=True)