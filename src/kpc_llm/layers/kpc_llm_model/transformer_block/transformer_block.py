from torch import nn,Tensor,manual_seed
from kpc_llm.layers.kpc_llm_model.transformer_block.multi_head_attention import MultiHeadAttention
from kpc_llm.layers.kpc_llm_model.kpc_norml import KpcNormal
from kpc_llm.layers.kpc_llm_model.transformer_block.feed_forward import KpcFeedForward

KpcTransformerBlockCfg=  {
    "vcab_sz" : 50524,
    "cntext_lnth" : 8,
    "emb_dim" : 512,
    "n_heads" : 9,
    "n_trnsfmr_layers" : 6,
    "drop_rt" : 0.5,
    "qkv_bias" : False
}


class KpcTransformerBlock(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.drop = nn.Dropout(cfg["drop_rt"])
        self.attention= MultiHeadAttention()
        # normal 1 2它们各自拥有独立的、需要学习的参数（Weights 和 Bias）,归属于不同的学习层，所以必须要两个，
        self.normal1 = KpcNormal(cfg["emb_dim"])
        self.normal2 = KpcNormal(cfg["emb_dim"])
        self.ff = KpcFeedForward(embed_dim=cfg["emb_dim"])

    def forward(self,x):
        # 1 注意力部分 
        # 残差链接，快捷链接保存的初始x值
        shortcut_x = x
        # 这里的必须是 带batch的 embedding 过的 token数据。数据进入前必须normal 统一数据空间规格
        x = self.normal1(x)
        x = self.attention(x)
        # 这里的上下文向量 需要drop保证泛化
        x = self.drop(x)
        # 残差链接最后输出
        x = shortcut_x + x

        # 2 前向传播部分
        shortcut_x2 = x
        x = self.normal2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = shortcut_x2 + x
        return x


