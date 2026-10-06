""" 固定的Training config数据类 """

from dataclasses import dataclass

# slots=True 锁死了属性名称不可更改，不可添加
@dataclass(slots=True)
class TrainConfig():
    cnf_name:str = "default"
    vcab_sz: int = 100277
    cntext_lnth: int =  256
    # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
    max_cntxt_pstion_lnth: int = 256
    emb_dim: int = 512
    heads_num: int = 8
    trnsf_blocks_num: int = 8
    drop_rt: float = 0.1
    qkv_bias :bool =  False
    batch_size: int = 16
    # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
    returnSoftmax :bool =  False


# gpt2 tiktokenizer cl100k_base vocab_size 100277 cnf
GPT2_cl100k_base_CNF =TrainConfig (
    cnf_name="GPT2_cl100k_base_CNF",
    vcab_sz=100277,
    cntext_lnth=256,
    max_cntxt_pstion_lnth=256,
    emb_dim=512,
    heads_num=8,
    trnsf_blocks_num= 8,
    batch_size=16
)