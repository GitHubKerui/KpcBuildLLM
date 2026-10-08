"""
固定的 Training config 数据类。

本项目的字段用缩写命名，与 GPT-2 官方命名的对应关系：
    emb_dim               -> n_embd
    heads_num             -> n_head
    trnsf_blocks_num      -> n_layer
    vcab_sz               -> vocab_size
    cntext_lnth           -> n_positions（训练时使用的上下文长度）
    max_cntxt_pstion_lnth -> n_positions（位置嵌入表长度）
"""

from dataclasses import asdict, dataclass, fields, replace


# slots=True 锁死属性名：不能改名，也不能动态添加字段
@dataclass(slots=True)
class TrainConfig:
    cnf_name: str = "default"
    vcab_sz: int = 100277
    cntext_lnth: int = 256
    # 位置嵌入表长度，用于生成位置 id，必须 >= cntext_lnth
    max_cntxt_pstion_lnth: int = 256
    emb_dim: int = 512
    heads_num: int = 8
    trnsf_blocks_num: int = 8
    drop_rt: float = 0.1
    qkv_bias: bool = False
    batch_size: int = 16
    # 生成时返回 softmax 概率还是 logits 的 argmax（即 tokenid）
    returnSoftmax: bool = False

    # -----------------------------------------------------------------------
    # 派生属性：没有类型注解，因此不是 dataclass 字段，
    # asdict() 的 key 集合保持与原字段列表完全一致，
    # 现有消费方（KpcLLMModel / TransformerBlock / hardware_require_calcu）
    # 可以继续把 asdict(cnf) 当纯 dict 用。
    # -----------------------------------------------------------------------

    @property
    def head_dim(self) -> int:
        """单个注意力头的维度。GPT-2 四个规格全是 64。"""
        return self.emb_dim // self.heads_num

    @property
    def ffn_dim(self) -> int:
        """
        FFN 中间层维度。feed_forward.py 里硬编码为 emb_dim*4，
        而 GPT-2 四个规格的官方 n_inner 恰好都等于 4*n_embd，所以无需新增字段。
        """
        return self.emb_dim * 4

    # -----------------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------------

    def validate(self) -> None:
        """训练前自检。尽早失败，避免跑完一个 epoch 才发现维度对不上。"""
        if self.emb_dim <= 0 or self.heads_num <= 0 or self.trnsf_blocks_num <= 0:
            raise ValueError(f"[{self.cnf_name}] emb_dim/heads_num/trnsf_blocks_num 必须为正整数")
        if self.emb_dim % self.heads_num != 0:
            raise ValueError(
                f"[{self.cnf_name}] emb_dim({self.emb_dim}) 必须能被 heads_num({self.heads_num}) 整除，"
                f"否则 MultiHeadAttention 的 reshape 会失败"
            )
        if self.cntext_lnth > self.max_cntxt_pstion_lnth:
            raise ValueError(
                f"[{self.cnf_name}] cntext_lnth({self.cntext_lnth}) 不能大于 "
                f"max_cntxt_pstion_lnth({self.max_cntxt_pstion_lnth})，否则位置嵌入表不够长"
            )
        if not 0.0 <= self.drop_rt < 1.0:
            raise ValueError(f"[{self.cnf_name}] drop_rt({self.drop_rt}) 必须落在 [0, 1) 区间")
        if self.batch_size <= 0:
            raise ValueError(f"[{self.cnf_name}] batch_size({self.batch_size}) 必须为正整数")

    def estimate_params(self, attn_out_proj: bool = True, tied_output: bool = False) -> int:
        """
        估算参数量，用于训练前判断显存是否够。

        默认参数按**本项目实际实现**估算，与 KpcLLMModel 的真实参数量零误差：
            attn_out_proj=True -> 注意力建 Q/K/V/O 四个投影
            tied_output=False  -> out_liner 独立，不与 token embedding 共享权重
        传 tied_output=True 切到 GPT-2 官方口径（输出层与 token embedding 绑定），
        结果与官方标称的 124M/355M/774M/1558M 吻合到 1% 以内。

        结构拆解：
            token/位置 embedding
            + 每层[Q,K,V(,O) 投影(含 bias) + FFN 两层(含 bias) + 2 个 LayerNorm]
            + 末尾 final_norml
            + 输出层 out_liner（tied_output=True 时不计）
        """
        d, ffn = self.emb_dim, self.ffn_dim
        # token embedding + 位置 embedding
        emb = self.vcab_sz * d + self.max_cntxt_pstion_lnth * d
        # 每个投影含 weight 与 bias
        n_proj = 4 if attn_out_proj else 3
        attn = n_proj * d * d + n_proj * d
        ffn_layer = 2 * d * ffn + ffn + d
        # 每层 2 个归一化层，每个含 weight+bias 共 2*d，合计 4*d（别写成 2*d）
        norm = 2 * 2 * d
        total = emb + self.trnsf_blocks_num * (attn + ffn_layer + norm) + 2 * d
        # 输出层；tied_output=True 时与 token embedding 共享权重，不重复计入
        if not tied_output:
            total += self.vcab_sz * d + (self.vcab_sz if self.qkv_bias else 0)
        return total


# ===========================================================================
# 预置配置
# ===========================================================================

# 本项目默认：tiktoken cl100k_base 词表(100277)，小尺寸，用于快速跑通流程
GPT2_cl100k_base_CNF = TrainConfig(
    cnf_name="GPT2_cl100k_base_CNF",
    vcab_sz=100277,
    cntext_lnth=256,
    max_cntxt_pstion_lnth=256,
    emb_dim=512,
    heads_num=8,
    trnsf_blocks_num=8,
    batch_size=16,
)

# ===========================================================================
# GPT-2 官方四个规格
#
# 对齐官方取值时有两个容易踩的坑：
# 1) qkv_bias=True —— GPT-2 用 Conv1D 实现 Q/K/V，带 bias；
#    TrainConfig 默认是 False，照抄默认值会和官方权重结构对不上。
# 2) 不需要新增 FFN 字段 —— feed_forward.py 里 FFN 硬编码为 emb_dim*4，
#    而四个规格的官方 n_inner(3072/4096/5120/6400) 恰好都等于 4*n_embd。
#
# 四个规格的 head_dim 都是 64，这是 GPT-2 系列的设计约定。
# vcab_sz=50257 是 GPT-2 自带的 BPE 词表；若改用本项目 tiktoken cl100k_base 需改成 100277。
# ===========================================================================

GPT2_SMALL_CNF = TrainConfig(
    cnf_name="GPT2_SMALL_CNF",
    vcab_sz=50257,
    cntext_lnth=1024,            # 官方 n_positions
    max_cntxt_pstion_lnth=1024,
    emb_dim=768,                 # n_embd
    heads_num=12,                # n_head
    trnsf_blocks_num=12,         # n_layer
    drop_rt=0.1,                 # 官方 resid/attn/embd 三处 dropout 都是 0.1
    qkv_bias=True,
    batch_size=4,                # 按本项目显存量级保守取值
)

GPT2_MEDIUM_CNF = TrainConfig(
    cnf_name="GPT2_MEDIUM_CNF",
    vcab_sz=50257,
    cntext_lnth=1024,
    max_cntxt_pstion_lnth=1024,
    emb_dim=1024,                # n_embd, head_dim=64
    heads_num=16,                # n_head
    trnsf_blocks_num=24,         # n_layer
    drop_rt=0.1,
    qkv_bias=True,
    batch_size=2,                # 层数和维度都涨了，batch 相应减半
)

GPT2_LARGE_CNF = TrainConfig(
    cnf_name="GPT2_LARGE_CNF",
    vcab_sz=50257,
    cntext_lnth=1024,
    max_cntxt_pstion_lnth=1024,
    emb_dim=1280,                # n_embd, head_dim=64
    heads_num=20,                # n_head
    trnsf_blocks_num=36,         # n_layer
    drop_rt=0.1,
    qkv_bias=True,
    batch_size=1,                # 单卡能跑起来的保守值
)

GPT2_XL_CNF = TrainConfig(
    cnf_name="GPT2_XL_CNF",
    vcab_sz=50257,
    cntext_lnth=1024,
    max_cntxt_pstion_lnth=1024,
    emb_dim=1600,                # n_embd, head_dim=64
    heads_num=25,                # n_head
    trnsf_blocks_num=48,         # n_layer
    drop_rt=0.1,
    qkv_bias=True,
    batch_size=1,
)

# 规格名 -> 配置实例。键名与 HuggingFace 上 gpt2 系列的命名后缀一致
GPT2_CNFS = {
    "small": GPT2_SMALL_CNF,
    "medium": GPT2_MEDIUM_CNF,
    "large": GPT2_LARGE_CNF,
    "xl": GPT2_XL_CNF,
}


def get_gpt2_cnf(size: str = "small", **overrides) -> TrainConfig:
    """
    按规格名取 GPT-2 配置，可就地覆盖任意字段。

    Args:
        size: 规格名。支持 "small" / "gpt2-medium" / "GPT2_XL" / "gpt2-large-774M.pth"
              等写法，内部做子串匹配后归一化到 GPT2_CNFS 的键。
        **overrides: 要覆盖的字段，例如 cntext_lnth=512、drop_rt=0.0。

    Returns:
        新的 TrainConfig 实例。内部用 replace 做浅拷贝，
        因此改返回值不会污染 GPT2_CNFS 里的全局常量。

    Raises:
        KeyError: size 匹配不到任何已知规格。
        TypeError: overrides 里出现了 TrainConfig 不存在的字段名。
    """
    key = str(size).lower()
    # "xl" 放最后判断，否则 "large" 之类的写法可能被误伤
    for name in ("small", "medium", "large", "xl"):
        if name in key:
            key = name
            break
    else:
        raise KeyError(f"未知的 GPT-2 规格 {size!r}，可选: {list(GPT2_CNFS)}")

    cnf = replace(GPT2_CNFS[key])
    valid = {f.name for f in fields(TrainConfig)}
    for field_name, value in overrides.items():
        if field_name not in valid:
            raise TypeError(f"TrainConfig 没有字段 {field_name!r}，可选: {sorted(valid)}")
        setattr(cnf, field_name, value)
    # 覆盖后可能出现 emb_dim/heads_num 不整除，这里顺带校验一次。
    # 注：basedpyright 对 @dataclass(slots=True) 解析不到类内自定义方法（误报，运行时正常）
    cnf.validate()  # type: ignore[attr-defined]
    return cnf
