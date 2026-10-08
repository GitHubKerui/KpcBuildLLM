import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import tiktoken
from kpc_llm.layers.kpc_llm_model.test.second_implement.llm_kpc2 import GPT2LMModel

class LitGPT2Reason(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = GPT2LMModel(cfg)

    def forward(self, x):
        return self.model(x)

    # ... training_step, validation_step, configure_optimizers 等保持不变 ...

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: int | None = None,
        enc_name: str = "cl100k_base"
    ) -> str:
        """
        接收输入字符串 prompt，自回归生成文本并返回解码后的字符串。
        
        Args:
            prompt: 提示词文本
            max_new_tokens: 最大新生成 token 数量
            temperature: 采样温度（>0），越高越随机，越低越确定
            top_k: 截断保留概率最高的 top-k 个候选词；若为 None 则全词表采样
            eos_id: 终止符 token id，遇到则提前停止生成
            enc_name: tiktoken 编码器名称
        """
        # 1. 确保模型处于评估模式
        self.eval()
        device = next(self.parameters()).device
        
        # 2. 分词与 Tensor 构造
        enc = tiktoken.get_encoding(enc_name)
        input_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        
        # 形状: [1, seq_len]
        idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        # 3. 自回归循环生成
        for _ in range(max_new_tokens):
            # 窗口裁剪：如果输入长度超过了位置嵌入长度，仅保留最新窗口
            idx_cond = idx[:, -self.cfg.max_cntxt_pstion_lnth:]
            
            # 前向计算 Logits: [1, cond_len, vocab_size]
            logits = self(idx_cond)
            
            # 仅提取最后一个时间步的预测: [1, vocab_size]
            logits = logits[:, -1, :]

            # 温度缩放（Temperature Scaling）
            if temperature > 0.0:
                logits = logits / temperature

                # Top-K 过滤：将前 top_k 之外的 logits 设为负无穷
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 小于 top_k 中最小值的 logits 全部置为 -inf
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Softmax 归一化为概率分布并进行多项式采样
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # temperature=0 时退化为贪婪采样（Greedy Search）
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # 遇到结束符终止
            if eos_id is not None and idx_next.item() == eos_id:
                break

            # 将新预测的 token 拼接到末尾
            idx = torch.cat((idx, idx_next), dim=1)

        # 4. 解码为文本字符串
        generated_token_ids = idx[0].tolist()
        return enc.decode(generated_token_ids)