""" 
1 验证 multinomial按照真实概率分布去抽样，而不是用argmax只采用贪心策略选最高概率index
增加生成字符的多样性。

2 但是采取multinomial策略后为，有一定概率采样出不合语法的字符的可能性，需要排除
某些低概率采样值，这时候采取截取一部分高概率分布来做整体分布来采样的策略。简称top-k

"""
import random
import re
from struct import unpack_from
from torch import Tensor, argmax, tensor,multinomial,bincount,manual_seed,topk,where
from kpc_llm.utils.bar_plot import plot_grouped_bar
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
# sample 是无放回的随机。是均匀分布采样
# choice 是放回的随机。是均匀分布采样
smpl_rdm = random.sample([ x/100 for x in range(100,1000)],9)
smpl_rdm_sf = tensor(smpl_rdm).softmax(-1)

logger.info(f"random_sample : {smpl_rdm}")
logger.info(f"random_sample softmax: {smpl_rdm_sf}")

# uniform的精度是长精度的，ai中使用这个。
unifrm_rdm: list[float] = [ round(random.uniform(1.00,10.00),2) for i in range(9) ] 
unifrm_rdm_sf = tensor(unifrm_rdm).softmax(-1)

logger.info(f"random_uniform : {unifrm_rdm}")
logger.info(f"softmax random_uniform : {unifrm_rdm_sf}")

# 用argmax始终会取最大值的index
max_idx = argmax(tensor(smpl_rdm))
logger.info(f"random_sample max_idx : ({max_idx},{smpl_rdm[max_idx]})")
max_idx2 = argmax(tensor(unifrm_rdm))
logger.info(f"random_sample max_idx2 : ({max_idx2},{unifrm_rdm[max_idx2]})")

"""测试multinomial的真实分布抽样100次,replacement=True表示有放回抽样，抽样样本被重新replace，当第二个参数采样数量大于样本数量时候，必须replacement=true。否则是无放回采样"""
mtnmil_smpls100 = multinomial(smpl_rdm_sf,100,replacement=True)
mtnmil_smpls100_2 =  multinomial(unifrm_rdm_sf,100,replacement=True)

logger.info(f"samples100 : {mtnmil_smpls100}")
logger.info(f"samples100_2 : {mtnmil_smpls100_2}")

""" 测试bincount 获取抽样词频的统计"""
bincount_re = bincount(mtnmil_smpls100,minlength=len(smpl_rdm_sf)).tolist()
bincount_re_2 = bincount(mtnmil_smpls100_2,minlength=len(unifrm_rdm_sf)).tolist()

logger.info(f"bincount_re : {bincount_re}")
logger.info(f"bincount_re_2 : {bincount_re_2}")

# 词频统计数据和分数logits合成元祖tuple
tuple_vcb_lgt = list(zip(bincount_re,smpl_rdm_sf.tolist()))
tuple_vcb_lgt_2 = list(zip(bincount_re_2,unifrm_rdm_sf.tolist()))

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


""" 测试同一组数据的不同的 temperature 带来的分布的均匀化 和 尖峰化 uniformly ，peaky """
def test_temperature(x_tensor:Tensor,sf_dim=-1,temperature=1.0):
     x_tensor = x_tensor/temperature
     x_sf = x_tensor.softmax(sf_dim)
     print(f"x_tensor : {x_tensor} ,temperature : {temperature}")
     print(f"x_sf : {x_sf}")
     return x_sf

test_temperature(tensor(smpl_rdm))
test_temperature(tensor(smpl_rdm),temperature=0.5)
test_temperature(tensor(smpl_rdm),temperature=2)

""" 测试top_k """



# 画柱状图
group_names =["temperature 0.5","temperature 1","temperature 1.5"]


# 概率分布中只抽取排名前top-k的概率,返回一个topk的元祖，封装了数据和index
random_sample_top4_logits,top4_index = topk(tensor(smpl_rdm),4) 
# 取topk最小值
logger.info("----------------------------------取topk最小值----------------------------------------------------")
logger.info(random_sample_top4_logits.shape)
topmin = random_sample_top4_logits[-1]
logger.info(topmin.shape)
logger.info(topmin)
# topk最小值以下都赋0
random_sample = tensor(smpl_rdm)
# 满足条件 取值来自 1，tensor(float("-inf")) 不满足条件 取值来自 2random_sample
random_sample =  where(topmin>random_sample,tensor(float("-inf")).to(random_sample.device),random_sample)

# 获取top4范围的vac
val_names = [verse_vocab[ele] for ele in top4_index.tolist()]
# 获取temperature调整三次后的3组概率分布,这个分布已经是top4之后，其他设置成 tensor(float("-inf"))
datasettmpture2 = [ test_temperature(random_sample,temperature=(i+1)*0.5) for i in range(3)]
# 从9个样本类型调整到前4的四类型分布后，放回抽样200次
dataset3 = [ multinomial(ds,200,replacement=True) for ds in datasettmpture2]
# 用bincount 统计3组概率分布抽样结果
bincount_re_3 = [bincount(ele,minlength=len(verse_vocab)).tolist() for ele in dataset3]
# logger.info("--------------------------------------------------------------------------------------")
# logger.info(val_names)
logger.info("--------------------------------------------------------------------------------------")
logger.info(bincount_re_3)


plot_grouped_bar(group_names,list(vocab.keys()),dataset=bincount_re_3)


