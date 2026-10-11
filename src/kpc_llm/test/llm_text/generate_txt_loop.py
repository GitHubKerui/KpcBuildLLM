"""
根据需要生成的字符的多少，循环生成字符的简单方法 
"""

from torch import no_grad,Tensor,softmax,cat

def generate_txt_loop(input_token_ids:Tensor,context_length,llm,max_generate_length=20):
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
        # 这里获取的词表的logits，所以需要转成概率分布。
        last_prediction_probs = softmax(last_prediction_logits,dim=-1)
        # 然后从分布中获取最大的tokenid,这里暂时用贪心算法方式,argmax 返回的是最大值的index值,
        # tensor.max返回的是最大值和index这里不一样
        last_prediction_tokenid = last_prediction_probs.argmax(dim=-1,keepdim=True)
        # 把这个tokenid 再拼接到原输入的最后
        input_token_ids = cat([input_token_ids,last_prediction_tokenid],dim=-1)
    return input_token_ids
