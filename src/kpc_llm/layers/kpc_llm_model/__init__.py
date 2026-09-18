"""GPT 组件的统一出口。 / Unified package for GPT components.

T0.1 之前，这个项目里并行存在两套 GPT 实现：

* ``kpc_llm.models.gpt.GPTModel`` —— 完整且可用；
* 本包早期的 ``KpcGPTModel`` —— 残缺占位，并且自己又重复组合了一整套
  embeddings / blocks / head。

两者已归并：**唯一的真实实现是 ``GPTModel``**，本包退化为「配置字典 → GPTModel」
的适配层。``KpcGPTModel`` / ``KpcTransformerBlock`` / ``KpcFinalNormal``
这几个名字继续对外导出，以免打断既有引用。
"""

from kpc_llm.layers.kpc_llm_model.kpc_llm_config import KPC_GPT_CONFIG_124M
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcGPTModel
from kpc_llm.layers.kpc_llm_model.transformer_block.transformer_block import KpcTransformerBlock
from kpc_llm.layers.kpc_llm_model.transformer_block.multi_head_attention import MultiHeadAttention

__all__ = [
    "KPC_GPT_CONFIG_124M",
    "KpcGPTModel",
    "KpcTransformerBlock",
    "MultiHeadAttention"
]
