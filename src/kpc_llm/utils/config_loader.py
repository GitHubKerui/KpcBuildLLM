"""项目配置加载工具。

封装对 configs/project_config.toml 的读取，集中处理配置路径的定位，
避免各处重复拼 get_project_root() / tomllib 逻辑。
"""

from pathlib import Path

import tomllib

from kpc_llm.utils.prj_dirc_file_tools import get_project_root


class ConfigLoader:
    """读取并解析 configs/project_config.toml。

    目前配置只承载"不随任务变化"的全局项（如 accelerate 总保存目录）；
    与某次具体任务相关的子目录选择由调用方代码控制，不放配置文件。
    """

    # 相对项目根目录的配置文件路径
    CONFIG_REL_PATH = ("configs", "project_config.toml")

    def __init__(self, config_path: Path | None = None) -> None:
        # 允许单测/特殊场景注入自定义路径；默认锚定项目根目录
        self.config_path = config_path or (get_project_root() / Path(*self.CONFIG_REL_PATH))
        self._config: dict = {}

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def load(self) -> dict:
        """解析 toml 文件，返回完整配置字典（含 [training] 等各段）。"""
        with open(self.config_path, "rb") as f:
            self._config = tomllib.load(f)
        return self._config

    @property
    def training(self) -> dict:
        """返回 [training] 段；首次访问时自动加载文件。"""
        if not self._config:
            self.load()
        return self._config.get("training", {})

    # ------------------------------------------------------------------
    # 字段访问
    # ------------------------------------------------------------------
    @property
    def accelerate_save_dir(self) -> str:
        """accelerate 总保存目录（相对项目根，可含多级路径）。

        只暴露总目录：其下按子目录分列不同任务/模型的存档，
        具体用哪个子目录由调用方（如 eff_calcu_train_loop 的 SAVE_SUBDIR）决定。
        """
        return str(self.training.get("accelerate_save_dir", "") or "")
