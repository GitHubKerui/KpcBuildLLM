from decimal import ROUND_HALF_UP, Decimal
import random
from kpc_llm.utils.logger import getlogger

logger = getlogger()


# 精确数学四舍五入
def safe_round(num:float,digits:int):
    d = Decimal(num)
    format = Decimal("0." + "0"*digits)
    return float(d.quantize(format,rounding=ROUND_HALF_UP))



""" 
测试 random.sample无放回采样，不会重复
"""
random_sample = random.sample(range(1,101),10)
logger.info(f"random_sample : {random_sample}")


""" 
测试 random.uniform 均匀采样，uniform是均匀分布的意思。可能重复 
"""
# uniform 以均匀分布采样小数，小数未阶段，这种小数叫尖刺，用round磨圆叫cut，3.1415用round保留3位，是3.141，不是传统的四舍五入3.142.
# uniform 以均匀分布采样小数，小数未阶段，这种小数叫尖刺，用round磨圆叫cut，
# 3.1415用round保留位数，这种round在精确计算4舍5入时候不可用。
random_uniform_single = random.uniform(3.1415,3.1419)
logger.info(f"random_uniform_single : {random_uniform_single}")


""" 
测试round的四舍五入，测试表明基本上跟数学上的四舍五入是一样的，好像不像传闻的round不可靠 
"""

round_num = round(2.665,2)
logger.info(f"round  2.665 : {round_num}")
logger.info(f"safe round 2.665 : {safe_round(round_num,3)}")

round_num = round(2.675,2)

logger.info(f"round  2.675 : {round_num}")
logger.info(f"safe round 2.675  : {safe_round(round_num,3)}")

round_num = round(3.1415,3)

logger.info(f"round  3.1415 : {round_num}")
logger.info(f"safe round 3.1415  : {safe_round(round_num,3)}")

round_num = round(3.1425,3)

logger.info(f"round  3.1425 : {round_num}")
logger.info(f"safe round 3.1425  : {safe_round(round_num,3)}")

round_num = round(3.1416,3)

logger.info(f"round  3.1416 : {round_num}")
logger.info(f"safe round 3.1416  : {safe_round(round_num,3)}")


f""" 测试用怎么实现 2.1显示成2.10 """

random_uniform: list[float] = [ round(random.uniform(1.00,10.00),2) for i in range(10) ]
# 统一成保留2位小数的格式显示 用"f{ :.2f}"，这样有些 2.1这种数就显示成2.10
random_uniform_2 = [f"{x:.2f}" for x in random_uniform]

logger.info(f"random_uniform : {random_uniform}")
logger.info(f"random_uniform_2 : {random_uniform_2}")

""" 测试random.randint 这里的范围选择是闭区间"""
# randint_num : [3, 2, 2, 1, 1, 2, 1, 1, 3, 1]
randint_num = [random.randint(1,3) for i in range(10)]
logger.info(f"randint_num : {randint_num}")

""" 测试random.choice """
# 从1-100里准备10个数
randint_num_ready = [random.randint(1,100) for i in range(10)]
logger.info(f"randint_num_ready : {randint_num_ready}")
# 随机取1个数，取三次
random_choice_num = [random.choice(randint_num_ready) for i in range(3)]
logger.info(f"random_choice_num : {random_choice_num}")

""" 测试random.shuffle """
logger.info(f"pre randint_num_ready : {randint_num_ready}")
# 这里不是赋值给新的打乱数组，是原来的数据指针不变
random.shuffle(randint_num_ready)
logger.info(f"shuffle randint_num_ready : {randint_num_ready}")
