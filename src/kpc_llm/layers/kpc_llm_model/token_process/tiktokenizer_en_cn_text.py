import tiktoken
# from transformers import AutoTokenizer

# 151936 词表规模，注意cnf配置文件对齐词表，179M参数规模，最优（≈1.5 字/token）中文压缩率，中文质量最好
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
text = " 测试 测试 测试 "
# ids = tokenizer.encode(text, add_special_tokens=False)   # 注意必须 False
# 1. 准备一段测试中文文本
text_zh = "每一次努力都会让你改变。"

# 2. 加载三种不同的内置词表 这是三个都是GPT的
enc_gpt2 = tiktoken.get_encoding("gpt2")          # GPT-2 词表 50257  (大小: ~5万 400mb) 纯英文更好，中文稍差
enc_gpt4 = tiktoken.get_encoding("cl100k_base")   # GPT-3.5/4 词表 100277 (大小: ~10万 800mb) 合适中英文
enc_gpt4o = tiktoken.get_encoding("o200k_base")   # GPT-4o 词表 200000 (大小: ~20万 1600mb) 合适中英文

# 3. 分别进行编码并计算 Token 数量
tokens_gpt2 = enc_gpt2.encode(text_zh)
tokens_gpt4 = enc_gpt4.encode(text_zh)
tokens_gpt4o = enc_gpt4o.encode(text_zh)

# 4. 打印结果
print(f"原始文本: '{text_zh}' (共 {len(text_zh)} 个字符)\n")
print(f"【GPT-2 词表】  Token 数量: {len(tokens_gpt2)} | Token IDs: {tokens_gpt2}")
print(f"【GPT-4 词表】  Token 数量: {len(tokens_gpt4)} | Token IDs: {tokens_gpt4}")
print(f"【GPT-4o 词表】 Token 数量: {len(tokens_gpt4o)} | Token IDs: {tokens_gpt4o}")