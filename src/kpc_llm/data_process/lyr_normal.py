'''
神经元的正则化，在输入或者输入网络前都做的正则化， 可是让训练过程更稳定，可以让训练权重更快更有效的收敛。
'''
from torch  import nn,Tensor,manual_seed,randn
from kpc_llm.utils.logger import getlogger

logger = getlogger()

class LayerNormal(nn.Moudle):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input):
        out = None
        return out


if __name__ == '__main__':
    manual_seed(517)
    test_tensor =  randn(4,6)
    logger.info(f'test_tensor before LayerNormal : {test_tensor}')
    
