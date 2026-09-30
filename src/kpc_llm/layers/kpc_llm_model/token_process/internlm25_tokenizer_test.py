"""
InternLM2.5-1.8B
含新增特殊 token 的总长度 vacab_size len(tokenizer): 92550
词表大小优秀 9.2 万 (极紧凑),英文能力优秀, 中文能力极强 (同档最优),最适合追求小词表、纯中英任务的微调
确保升级了 transformers 和 accelerate（用于管理显存部署）
pip install -U transformers accelerate torch
"""

from pathlib import Path

from pyprojroot import here
from transformers import AutoTokenizer
import sentencepiece as spm

# 官方 tokenizer.model 的第 354 号 piece 是真实字节 '\x00'，sentencepiece>=0.2.0 会拒绝加载。
# 这里改为加载 scripts/fix_internlm25_tokenizer.py 生成的本地修复副本（词表数量与 id 顺序不变）。
tokenizer_dir: Path = here() / "data" / "tokenizer" / "internlm2_5"
model_id = "internlm/internlm2_5-1_8b-chat"
# trust_remote_code 仍需保留：InternLM2Tokenizer 是自定义类（代码已落地本地，不再从 HF 拉取新版）
# use_fast=False：Fast 版是 transformers 自动转换的，decoder 会在每个 piece 前补空格
# （"深度 学习 与 ..."、"Ex ponential"），训练虽然只用 id 不受影响，但生成/还原会错。

# tokenizer = AutoTokenizer.from_pretrained(
#     str(tokenizer_dir),
#     trust_remote_code=True,
#     use_fast=False,
# )

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False,
)

print(f"tokenizer 类型: {type(tokenizer).__name__}")
print(f"InternLM2.5 词表大小 (Vocab Size): {tokenizer.vocab_size}")
print(f"含新增特殊 token 的总长度 len(tokenizer): {len(tokenizer)}")
print(f"实际最大 token id: {max(tokenizer.get_vocab().values())}")

text = "深度学习与指数移动平均 EMA (Exponential Moving Average)"

# ---- 编码：不要自动加特殊 token ----
ids = tokenizer.encode(text, add_special_tokens=False)
print("ids          :", ids)
print("无特殊token还原:", repr(tokenizer.decode(ids)))
print("保留标点前空格 :", repr(tokenizer.decode(ids, clean_up_tokenization_spaces=False)))

# ---- 如果 id 里已经混了 <s>，decode 时过滤掉 ----
ids_with_bos = tokenizer.encode(text)
print("带BOS的ids   :", ids_with_bos)
print("过滤特殊token :", repr(tokenizer.decode(ids_with_bos, skip_special_tokens=True)))

# ---- 只看 piece 时，把 ▁ 还原成空格再拼 ----
pieces = tokenizer.tokenize(text)
print("pieces       :", pieces)
print("手动拼        :", repr("".join(p.replace("\u2581", " ") for p in pieces).strip()))

# ---- 直接用 sentencepiece（最干净）----
sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
ids = sp.Encode(text)                      # 不加任何特殊 token
print(sp.Decode(ids))                      # 深度学习与指数移动平均 EMA (Exponential Moving Average)