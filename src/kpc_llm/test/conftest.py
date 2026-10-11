"""Pytest global configuration. / pytest 全局配置。

T0.2 建立的测试基座之一。它做两件事：

1. 把仓库的 ``src/`` 布局显式加入 ``sys.path`` —— 保证无论从哪个工作目录
   运行 pytest，``kpc_llm`` 都能被稳定解析，不受 editable install 状态影响。
2. 固定全局随机种子 —— 让形状、数值类断言可复现，避免偶然失败掩盖真问题。
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个用例执行前重置随机种子，避免用例之间相互污染。 / Reset seed per test."""
    torch.manual_seed(0)
    yield
