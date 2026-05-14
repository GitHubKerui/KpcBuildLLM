import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

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
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
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

# 使用示例
if __name__ == '__main__':
    logger.debug("调试信息（开发用）")
    logger.info("普通信息（启动成功、流程正常）")
    logger.warning("警告（非错误但需注意）")
    logger.error("错误（业务异常）")
    logger.critical("严重错误（程序崩溃）")