

from torch import nn,Tensor,manual_seed,arange
from kpc_llm.utils.logger import getlogger
from kpc_llm.layers.kpc_gpt_model.config_model import KPC_GPT_CONFIG_124M as cnf_kpc
from kpc_llm.layers.kpc_gpt_model.gpt_trnsf_block import KpcTransformerBlock
from kpc_llm.layers.kpc_gpt_model.gpt_norml import KpcFinalNormal


class KpcGPTModel(nn.Module):
    def __init__(self,cnf) -> None:
        super().__init__()
        vcab_sz = cnf['vcab_sz']
        emb_dim = cnf['emb_dim']
        cntext_lnth = cnf['cntext_lnth']
        trnsf_lyrs = cnf['n_trnsfmr_layers']
        qkv_bias = cnf['qkv_bias']
        drop_rt = cnf['drop_rt']

        self.tkn_emb = nn.Embedding(vcab_sz,emb_dim)
        self.pstn_emb = nn.Embedding( cntext_lnth,emb_dim)
        self.drop = nn.Dropout(drop_rt)
        self.trnsf_lyrs = nn.Sequential(*[KpcTransformerBlock(cnf) for _ in range(trnsf_lyrs)])
        self.final_norml = KpcFinalNormal(cnf['emb_dim'])
        self.out_liner = nn.Linear(cnf['emb_dim'],cnf['vcab_sz'],bias=qkv_bias)

    def forward(self,input):
        btch,cntext_lnth =  x.shape
        input_tokens = self.tkn_emb(input)
        x = input_tokens + self.pstn_emb(arange(cntext_lnth,device=input.device))
        x = self.drop(x)
        x = self.trnsf_lyrs(x)
        x = self.final_norml(x)
        out = self.out_liner(x)
        return out