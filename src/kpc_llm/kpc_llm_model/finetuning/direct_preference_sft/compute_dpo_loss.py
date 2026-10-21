
import torch
import torch.nn.functional as F
from torch import Tensor, mean, sigmoid,log

def compute_dpo_loss_raw(loss_p_c:Tensor, loss_p_r:Tensor,loss_r_c:Tensor,loss_r_r:Tensor,b:float):
    """
    计算DPO损失
    必须输入的是都是 cross entropy loss，对数概率差。不能是原生概率差
    因为cross entropy loss 是 -logp，所以logp 相减要翻号 -(loss_p_c - loss_r_c) 变 loss_r_c - loss_p_c
    loss_policy_chosen = loss_p_c 
    loss_policy_rejected = loss_p_r
    loss_reference_chosen = loss_r_c
    loss_reference_rejected = loss_r_r
    """
             
    part_chosen =  loss_r_c - loss_p_c
    part_rejected = loss_r_r - loss_p_r
    sgm_idx = b * (part_chosen - part_rejected)
    part_sigmoid = sigmoid(sgm_idx)
    part_log  = log(part_sigmoid)
    loss =  mean(part_log) * (-1)
    return loss

def compute_dpo_loss(loss_p_c: Tensor, loss_p_r: Tensor, loss_r_c: Tensor, loss_r_r: Tensor, b: float):
    """我们的实现。入参是 NLL=-logp，所以 logp 相减要翻号。

    logit = β·[(logπθ(y_w)-logπref(y_w)) - (logπθ(y_l)-logπref(y_l))]
    """
    logit = b * ((loss_r_c - loss_p_c) - (loss_r_r - loss_p_r))
    return -F.logsigmoid(logit).mean()


def make_data(n: int = 8, seed: int = 0):
    """造 logp 数据，返回 (policy_chosen, policy_rejected, ref_chosen, ref_rejected)"""
    torch.manual_seed(seed)
    return tuple(-(torch.rand(n) * 20 + 5) for _ in range(4))


def trl_dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    """TRL 1.16 的默认路径（sigmoid + reverse_kl），对应 dpo_trainer.py:1383-1388。
    注意：1.16 已删掉 DPOTrainer.dpo_loss 静态方法，只能照抄这段。"""
    delta = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * delta).mean()


def compare(betas=(0.0, 0.1, 1.0)):
    """两套实现逐 beta 对比"""
    c, r, rc, rr = make_data()
    for beta in betas:
        mine_raw = compute_dpo_loss_raw(-c, -r, -rc, -rr, beta)  # 我们吃 NLL
        mine = compute_dpo_loss(-c, -r, -rc, -rr, beta)  # 我们吃 NLL
        trl = trl_dpo_loss(c, r, rc, rr, beta)  # TRL 吃 logp
        same = torch.allclose(mine, trl, atol=1e-6)
        print(f"beta={beta:<4} mine_raw={mine_raw:.6f}  mine={mine:.6f} trl={trl:.6f} {'OK' if same else 'MISMATCH'}")
        assert same


# if __name__ == "__main__":
#     compare()
