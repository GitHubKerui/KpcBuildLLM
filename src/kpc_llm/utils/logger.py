from typing import Any


import functools
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from importlib.metadata import version
import time

from numpy import empty

# ====================== 日志配置（企业级标准） ======================
def getlogger(name="kcp_build_llm"):
    # 1. 创建日志器
    # step1 创建未设置的日志对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 全局最低级别
    logger.handlers.clear()  # 避免重复打印
    logger.propagate = False

    # 2. 日志格式（时间 + 级别 + 文件名 + 行号 + 信息）
    log_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ====================== 输出 1：控制台打印 ======================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    # step2 设置控制台打印格式和日记级别
    logger.addHandler(console_handler)

    # ====================== 输出 2：文件输出（按天分割） ======================
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)  # 自动创建 logs 文件夹

    # 按天分割日志，保留 30 天
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="D",        # 按天切割
        backupCount=30,  # 保留 30 份
        encoding="utf-8",
        delay=True
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    # step3  设置文件按天分割处理命名
    logger.addHandler(file_handler)

    return logger

# ====================== 全局使用 ======================
# 初始化一次，全项目通用
logger = getlogger()

def print_pkgs_version(pkgs=None):
    """ 打印包的版本信息 
        pkgs: 包名列表，默认值为None，表示打印所有常用包的版本信息
    """
    if pkgs is None:
        pkgs =["matplotlib",
            "numpy",
            "tiktoken",
            "torch",
            "tensorflow", # For OpenAI's pretrained weights
            "pandas" ]
    for pkg in pkgs:
        if pkg != "":
            logger.info(f"{pkg} version : {version(pkg)}")

""" 打印输入输出"""
def log_fn_io(func):
    import inspect
    # 拿到函数的完整信息，签名
    func_reflect = inspect.signature(func)
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        logger.info(f"\n===== 🚀 🚀 🚀 开始执行函数: {func.__name__} 🚀 🚀 🚀 =====")
        print(f"👉 【输入参数】:")
        try:
            # 用func签名绑定参数名称和参数值
            bound_args = func_reflect.bind(*args,**kwargs)
            # 没输入的默认值也用上
            bound_args.apply_defaults()
            for p_name,p_value in bound_args.arguments.items():
                print(f"\n- 参数 ({str(p_name):<12}) - 参数类型 ({type(p_value).__name__}  -")
                print(f"\n- 参数值 {str(p_value):<12} -")
        except Exception as e:
            # 兜底方案：如果绑定失败（比如传参严重错误），降级打印原始数据
            if args:   print(f"   - 原始位置参数 (args)   : {args}")
            if kwargs: print(f"   - 原始关键字参数 (kwargs): {kwargs}")
            logger.exception(f"f{e}")
       
        # 执行原函数
        re: Any = func(*args,**kwargs)

        # 2. 打印输出和耗时
        print(f"【输出结果】: {re}👈 ")
        # 返回的变量的名称或者类型
        re_name_tp = func_reflect.return_annotation
        if re_name_tp != inspect.Signature.empty:
            re_name = re_name_tp.__name__   
        else:
            re_name = type(re).__name__
        print(f"\n- 返回对象 {str(re_name):<12} : -")    
        print(f"\n- 返回值 {str(re):<12} : -")    
        print(f"=========================================\n")
        return re
    return wrapper

""" 打印耗费时间"""
def log_fn_cost_time(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        # 执行原函数
        start_time = time.time()
        re: Any = func(*args,**kwargs)
        end_time = time.time()
        print(f"\n⏱️ {func.__name__ }【执行耗时】: {(end_time - start_time):.4f} 秒")
        print(f"=========================================\n")
        return re
    return wrapper

def log_model(model):
    """ 打印model 的大概情况"""
    # 用一个集合来记录已经打印过的参数类型后缀
    printed_types = set()

    for name, param in model.named_parameters():
        # 提取参数的通用类型后缀
        # 例如: "model.layers.0.self_attn.q_proj.weight" -> "self_attn.q_proj.weight"
        # 或者简单通过 "." 切分，取最后两段作为类型标识
        name_parts = name.split(".")
        param_type = ".".join(name_parts[-2:]) if len(name_parts) >= 2 else name

        if param_type not in printed_types:
            printed_types.add(param_type)
            
            # 打印该类型的代表性参数信息
            logger.info(
                f"参数类型: {param_type:<20} | "
                f"样例全称: {name:<35} | "
                f"形状: {str(param.shape):<26} | "
                f"梯度(Requires Grad): {str(bool(param.requires_grad)):<3} | "
                f"设备: {param.device}"
            )



# 使用示例
if __name__ == '__main__':
    logger.debug("调试信息（开发用）")
    logger.info("普通信息（启动成功、流程正常）")
    logger.warning("警告（非错误但需注意）")
    logger.error("错误（业务异常）")
    logger.critical("严重错误（程序崩溃）")