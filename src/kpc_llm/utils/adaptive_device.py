""" 自适应获取device，或者获取指定device """
import torch
from kpc_llm.utils import getlogger


logger = getlogger()
# 根据硬件自适应获取device
def get_adaptive_device():

    # 检验老黄的显卡
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Nvidia CUDA device is available")
        return device
    # MPS 全称是 Metal Performance Shaders，它是苹果公司专门为 Mac M1/M2/M3/M4 芯片 开发的 GPU 加速后端
    elif torch.backends.mps.is_available():
        # 0 号位是 '2' (主版本号 Major)
        # 1 号位是 '11' (次版本号 Minor)
        # 2 号位是 '0' (修补版本号 Patch)
        # 迭代器，用来把所有 ele转换成int整数数字，赋值给 major 和 minor 
        major,minor = map(int,torch.__version__.split(".")[:2]) 
        # 利用 Python 元组的对比特性，强行校验当前 PyTorch 的版本是否大于等于 2.9 版本。
        # 如果用户装的 PyTorch 版本低于 2.9，即便他的 Mac 支持 MPS，也不用，因为低版本的 MPS 不够稳定。只有高于等于 2.9 版本才能使用 
        if (major,minor) >= (2,9):
            device = torch.device("mps")
            logger.info("Apple MPS device is available")
            return device
    else:
        logger.info("No Nvidia CUDA or Apple MPS device is available, using CPU instead")
        return torch.device("cpu")


# 根据指定device获取device "cuda" or "mps" or "cpu"
def get_device(deviceName):

    if deviceName is not None or deviceName != "":
        # 检验老黄的显卡
        if torch.cuda.is_available() and deviceName == "cuda":
            device = torch.device("cuda")
            logger.info("Nvidia CUDA device is available")
            return device
        # MPS 全称是 Metal Performance Shaders，它是苹果公司专门为 Mac M1/M2/M3/M4 芯片 开发的 GPU 加速后端
        elif torch.backends.mps.is_available() and deviceName == "mps":
            # 0 号位是 '2' (主版本号 Major)
            # 1 号位是 '11' (次版本号 Minor)
            # 2 号位是 '0' (修补版本号 Patch)
            # 迭代器，用来把所有 ele转换成int整数数字，赋值给 major 和 minor 
            major,minor = map(int,torch.__version__.split(".")[:2]) 
            # 利用 Python 元组的对比特性，强行校验当前 PyTorch 的版本是否大于等于 2.9 版本。
            if (major,minor) >= (2,9):
                device = torch.device("mps")
                logger.info("Apple MPS device is available")
                return device
        elif deviceName == "cpu":
            # 专门要求用CPU
            logger.info("Using CPU device")
            return torch.device("cpu")
        else:
            # 如果指定的GPU不可用则自适应用GPU
            return  get_adaptive_device()
    else:
        return  get_adaptive_device()


# test
# device = get_adaptive_device()
# device = get_device("cuda")
# logger.info(f"device : {device}")