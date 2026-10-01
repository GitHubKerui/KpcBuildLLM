""" 
1 验证 multinomial按照真实概率分布去抽样，而不是用argmax只采用贪心策略选最高概率index
增加生成字符的多样性。

2 但是采取multinomial策略后为，有一定概率采样出不合语法的字符的可能性，需要排除
某些低概率采样值，这时候采取截取一部分高概率分布来做整体分布来采样的策略。简称top-k

"""
import random
import re

from torch import argmax, tensor,multinomial,bincount,manual_seed

from kpc_llm.utils.logger import getlogger

logger = getlogger()

#  模拟词表数据
vocab = {
     "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

# 词表翻转
verse_vocab = { v:k for k,v in vocab.items()}
logger.info(verse_vocab)

# 锁死random,和tensor
random.seed(825)
manual_seed(825)
# 模拟最后一个预测token的词表分数,这种方法的精度真的就是保留2位为止
random_sample = random.sample([ x/100 for x in range(100,1000)],6)
random_sample_sf = tensor(random_sample).softmax(-1)

logger.info(f"random_sample : {random_sample}")
logger.info(f"random_sample softmax: {random_sample_sf}")

# uniform的精度是长精度的，ai中使用这个。
random_uniform: list[float] = [ round(random.uniform(1.00,10.00),2) for i in range(6) ] 
random_uniform_sf = tensor(random_uniform).softmax(-1)

logger.info(f"random_uniform : {random_uniform}")
logger.info(f"softmax random_uniform : {random_uniform_sf}")

# 用argmax始终会取最大
argmax_get = argmax(tensor(random_sample))
logger.info(f"random_sample argmax_get : ({argmax_get},{random_sample[argmax_get]})")
argmax_get_2 = argmax(tensor(random_uniform))
logger.info(f"random_sample argmax_get : ({argmax_get_2},{random_uniform[argmax_get_2]})")

#测试multinomial的真实分布抽样100次,replacement=True表示有放回抽样，抽样样本被重新replace，当第二个参数采样数量大于样本数量时候，必须replacement=true。否则是无放回采样
samples100 = multinomial(random_sample_sf,100,replacement=True)
samples100_2 =  multinomial(random_uniform_sf,100,replacement=True)

logger.info(f"samples100 : {samples100}")
logger.info(f"samples100_2 : {samples100_2}")

# 测试 bincount 获取抽样词频的统计
bincount_re = bincount(samples100,minlength=len(random_sample_sf)).tolist()
bincount_re_2 = bincount(samples100_2,minlength=len(random_uniform_sf)).tolist()

logger.info(f"bincount_re : {bincount_re}")
logger.info(f"bincount_re_2 : {bincount_re_2}")

# 词频统计数据和分数logits合成元祖tuple
tuple_vcb_lgt = list(zip(bincount_re,random_sample_sf.tolist()))
tuple_vcb_lgt_2 = list(zip(bincount_re_2,random_uniform_sf.tolist()))

# 1 单词 2词频统计 3概率分数
vac_binc_re = [(verse_vocab[i], ele[0],ele[1]) for i,ele in enumerate(tuple_vcb_lgt)]
vac_binc_re_2 = [(verse_vocab[i], ele[0],ele[1]) for i,ele in enumerate(tuple_vcb_lgt_2)]

logger.info(f"vac_binc_re : {vac_binc_re}")
logger.info(f"vac_binc_re : {vac_binc_re_2}")
logger.info("--------------------------------------------------------------------------------------")
for ele in vac_binc_re:
    logger.info(ele)

logger.info("--------------------------------------------------------------------------------------")
for ele in vac_binc_re_2:
    logger.info(ele)