"""
根据需要生成的字符的多少，循环生成字符的简单方法 
"""

from torch import no_grad, Tensor, softmax, cat, topk, masked_fill

def generate_txt_loop(input_token_ids:Tensor,context_length,llm,max_generate_length=20,temperature:float=0.0,top_k:int=0,eos_id=None):
    # 根据长度循环生成后面的字
    for i in range(max_generate_length):
        # 如果输入长度小于context_length，则直接返回，否则截取最后context_length个token
        context_ids = input_token_ids[:,-context_length:]
        # 这里要关闭反向传播需要的梯度极端
        with no_grad():
            # 这里是生成了一个batch的预测，只有最后一个tokenid是预测tokenid
            prediction_ids = llm(context_ids)

        # 截取最后一个tokenid的词表的logits逻辑值
        last_prediction_logits = prediction_ids[:,-1]

        # 1 添加topk逻辑，这里放在temperature前面是合理的
        if top_k > 0 and last_prediction_logits is not None:
            last_prediction_logits = add_top_k(last_prediction_logits,top_k)

        # 2 添加temperature逻辑（注意：add_sftmax_temperature 返回的是概率，不是 logits）
        if temperature > 0.0 and last_prediction_logits is not None:
            last_prediction_probs = add_sftmax_temperature(last_prediction_logits,temperature)
            #这里添加多项式分布提取
            last_prediction_tokenid = last_prediction_probs.multinomial(1)
        # 3 否则直接贪心算法
        elif last_prediction_logits is not None:
            # 这里获取的词表的logits，所以需要转成概率分布。
            last_prediction_probs = softmax(last_prediction_logits,dim=-1)
            # 然后从分布中获取最大的tokenid,这里暂时用贪心算法方式,argmax 返回的是最大值的index值,
            # tensor.max返回的是最大值和index这里不一样
            last_prediction_tokenid = last_prediction_probs.argmax(dim=-1,keepdim=True)

        # 把这个tokenid 再拼接到原输入的最后
        input_token_ids = cat([input_token_ids,last_prediction_tokenid],dim=-1)
        # 如果指定结束tokenid 遇到后就结束输出。
        # 只判断本轮新生成的最后一个 token；不能写成 input_token_ids == eos_id，
        # 因为序列已经很长，整体比较会返回 bool tensor，触发
        # "Boolean value of Tensor with more than one element is ambiguous"
        if eos_id is not None and bool((input_token_ids[:,-1] == eos_id).all()):
            break


    return input_token_ids


def add_sftmax_temperature(x:Tensor,temperature:float=1):
    """按 temperature 缩放 logits 后转成概率分布。返回的是 probs，不是 logits。"""
    if temperature > 0.0:
        x_tmpt = x/temperature
        return softmax(x_tmpt,-1)
    else:
        return softmax(x,-1)

def add_top_k(x:Tensor,top_k):
    """
    top-k 屏蔽：把 logits 中排在第 top_k 名之后的值置为 -inf。
    x 形状: (..., vocab_size)   返回形状与 x 完全一致（仍是 logits）
    """
    # top_k 不能超过词表大小，否则 topk 会直接报错
    top_k = min(top_k, x.size(-1))
    if top_k <= 0:
        return x
    # topk 返回的是 tuple (values, indices)，必须解包取 values；
    # 写成 x_tpk = topk(...) 再 x_tpk[-1] 拿到的是 indices 而不是最小值
    x_tpk, x_tpk_ind = topk(x,top_k)
    # topk 默认按降序排列，最后一列即第 top_k 大的值；
    # 用 [..., -1:] 保留维度，才能和 x 广播对齐（写成 [-1] 会塌成 (batch,)）
    x_tpk_min = x_tpk[...,-1:]
    # masked_fill 会自动保持 x 的 dtype，比 where + tensor(float("-inf")) 更安全
    x_mask_tpk = x.masked_fill(x < x_tpk_min, float("-inf"))
    # 只返回屏蔽后的 logits，概率计算和采样交给调用方
    return x_mask_tpk
