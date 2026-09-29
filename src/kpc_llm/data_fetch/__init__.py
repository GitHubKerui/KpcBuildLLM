"""Data fetching utilities."""

# 不要在包的 __init__.py 里 import 子模块：
# kpc_llm.data_fetch.my_tokenizer 已不存在，模块级 import 会让整个包不可用。
# 需要时请显式导入，例如：
#   from kpc_llm.data_fetch.textloader import getTxtStr
