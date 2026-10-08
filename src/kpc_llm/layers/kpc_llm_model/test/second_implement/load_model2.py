import tiktoken
import torch
from kpc_llm.layers.kpc_llm_model.test.second_implement.generate_loop2 import LitGPT2Reason

# 1. 指定保存的最佳 ckpt 路径
ckpt_path = "checkpoints_GPT2_cl100k_base_CNF/gpt2-best-step=0450-val_loss=3.210.ckpt"

# 2. 加载模型（会自动恢复配置和网络架构权重）
model = LitGPT2Reason.load_from_checkpoint(ckpt_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 3. 获取 <|endoftext|> 对应的 ID（cl100k_base 中通常为 100257）
enc = tiktoken.get_encoding("cl100k_base")
eos_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

# 4. 执行生成
prompt = "Deep learning is"
output_text = model.generate_text(
    prompt=prompt,
    max_new_tokens=60,
    temperature=0.7,   # 0.7 平衡多样性与语法连贯性
    top_k=50,
    eos_id=eos_id
)

print("-" * 40)
print(output_text)
print("-" * 40)