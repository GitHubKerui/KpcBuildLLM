"""
Top-K 最优权重存档与早停判定（基于 Hugging Face Accelerate）。

Accelerate 本身**不提供**任何 topk 管理回调（不存在 Lightning `ModelCheckpoint`
那种带 monitor / save_top_k / mode 的机制），所以"判定 val_loss 是否最优、只保留
N 份、淘汰旧档"这三件事必须自己实现。但 Accelerate 提供了比 torch.save 更稳健的
存档原语，本模块用它落盘。
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from kpc_llm.utils.logger import getlogger

logger = getlogger()


class TopKSaver:
    """按 val_loss 维护最优的前 topk 份权重，并在连续 exit_n 次没改善时返回 True。

    **早停的两个防误杀开关**（都可选，默认关闭）：

        min_delta             —— 改善必须超过该阈值才算"刷新最优"。用于滤掉 val 噪声：
                                 val 抖动常常远大于真实改善，不加阈值时要么在噪声里
                                 反复归零（永不早停），要么被一次抖动锁死最优值而误杀。
        min_evals_before_stop —— 评估次数不到该值之前不允许早停。用于保证训练在早期
                                 （val 还很高、噪声最大）不被掐断，先跑够一定量再看。

    **为什么用 accelerator.save_model 而不是 torch.save**

        | | torch.save | accelerator.save_model |
        |---|---|---|
        | 格式 | pickle | safetensors 默认，无 pickle 反序列化风险 |
        | DDP 包装 | 需手动 unwrap，否则键带 module. 前缀 | 内部 get_model_state_dict 自动解包 |
        | 超大模型 | 单文件，超 2GB 有兼容问题 | max_shard_size 自动分片 |
        | 分布式 | 需自己判断 rank | 自动只让主进程写 |

    **safetensors 不支持共享张量**（shared tensors），模型若有权重绑定会直接报错。
    本项目 KpcLLMModel 实测：101 个 state_dict 张量对应 101 个唯一 storage，完全无
    共享，因此 safe_serialization=True 可安全使用。若将来启用 tied weights
    （out_liner 与 token embedding 绑定），save_model 会报错，需要改回 torch.save。

    **两条状态持久化路径，分工不同、互为补充**

        1. topk_meta.json —— topk 目录自包含，不依赖 accelerate 存档。训练在 epoch
           中途被中断（还没到 save_state 时机）时靠它恢复 topk 列表。
        2. state_dict() / load_state_dict() —— 供 register_for_checkpointing 注册，
           跟着 accelerator.save_state() 一起存成 custom_checkpoint_0.pkl，断点续训
           时自动带回，早停计数也随之恢复。

    **安全提醒**：accelerate 的 load_custom_state 源码明确写了
    "Will always set weights_only=False"，即读取 custom_checkpoint_*.pkl 会禁用
    pickle 保护。因此**断点存档目录必须是自己本地产生的**，不要接受外部来源的存档
    （与 model_save_load_test.py 里 weights_only=True 那段安全注释是同一个道理）。

    **单卡限制**：本类不做 DDP 通信。 Accelerate 的主场景是多卡，但多卡下
    evaluate_model 算的是"局部均值"（它没有做 gather_for_metrics / reduce），
    各进程数值不同会让最优判定错乱。要上多卡，必须先给验证 loss 加全局聚合。
    """

    def __init__(self, accelerator, topk: int = 3, exit_n: int = 10,
                 root_dir: str = "checkpoints/topk_safetensors",
                 modelname: str = "kpcModel",
                 min_delta: float = 0.0,
                 min_evals_before_stop: int = 0):
        # topk<=0 会让"门槛"永远取不到有效值；exit_n<=0 则是第一次评估就判定该停，
        # 等于训练完全不跑。min_delta/min_evals_before_stop 为负则语义不成立。
        # 这些都要在构造期就拦住。
        if topk <= 0 or exit_n <= 0:
            raise ValueError(f"topk/exit_n 必须为正整数，收到 topk={topk}, exit_n={exit_n}")
        if min_delta < 0:
            raise ValueError(f"min_delta 不能为负，收到 {min_delta}")
        if min_evals_before_stop < 0:
            raise ValueError(f"min_evals_before_stop 不能为负，收到 {min_evals_before_stop}")
        self.accelerator = accelerator
        self.topk, self.exit_n = topk, exit_n
        # 改善阈值：val_loss 要比历史最优再低至少这么多，才算"刷新最优"
        self.min_delta = min_delta
        # 早停门槛：评估次数不到这个数之前一律不允许早停
        self.min_evals_before_stop = min_evals_before_stop
        self.modelname = modelname
        # 用 parents=True 一步建到位：项目里 create_dir_under_prj 的 mkdir 不带
        # parents，多级路径会 FileNotFoundError
        self.root_dir = Path(root_dir) / modelname
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root_dir / "topk_meta.json"

        # entries: [(val_loss, 目录名)]，按 val_loss 升序，entries[0] 即当前最优
        self.entries: list[tuple[float, str]] = []
        self.best_val_loss = float("inf")     # inf 表示还没评估过，任何 loss 都能刷新它
        self.no_improve_count = 0
        self.eval_count = 0                   # 已评估次数，用于 min_evals_before_stop 门槛
        self._restore()

        # 注册后，accelerator.save_state() 会自动把本对象存成
        # {save_dir}/custom_checkpoint_0.pkl，load_state() 时自动恢复。
        # 所以 state_dict/load_state_dict 必须实现，且返回普通 Python 对象。
        self.accelerator.register_for_checkpointing(self)

    # -----------------------------------------------------------------------
    # 对外接口
    # -----------------------------------------------------------------------

    def step(self, model, train_loss, val_loss, epoch_n, batch_n, seen_token_n) -> bool:
        """走一步：必要时存一份权重，并返回是否该停止训练。

        Args:
            model:        要保存的模型；内部会 unwrap_model，DDP 包装也能安全存
            train_loss:   本轮训练集 loss，只写进日志/文件名供追溯，不参与排名
            val_loss:     本轮验证集 loss，排名和早停都只看它
            epoch_n / batch_n / seen_token_n: 当前进度，写进目录名便于事后定位

        Returns:
            True  -> 满足早停条件（连续 exit_n 次没改善，且已跑够最小评估次数）
            False -> 继续训练
        """
        # 0) 评估次数计数：早停只在评估次数达到 min_evals_before_stop 之后才允许触发
        self.eval_count += 1

        # 1) 早停计数：改善必须超过 min_delta 才算刷新历史最优（用 < 而非 <=，
        #    相等时算"没改善"）。加 min_delta 是为了滤掉 val 噪声：本项目 val 抖动
        #    约 0.3，而真实改善常只有 1e-3 量级，不加阈值极易被一次抖动"锁死"最优值，
        #    导致之后连续不刷新而被误杀。
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # 2) 排进前 topk 才存盘。未满 topk 时门槛是 inf，第一次必存；满了之后
        #    只有比"第 topk 名"更好才值得存，否则写进去立刻就被淘汰，白费一次磁盘写。
        #    注意：排名/存盘只看真实 val_loss，不套 min_delta，保证真最优一定被存下。
        threshold = self.entries[-1][0] if len(self.entries) >= self.topk else float("inf")
        if val_loss < threshold:
            self._save_and_rank(model, train_loss, val_loss, epoch_n, batch_n, seen_token_n)

        # 3) 早停判定：既要"连续 exit_n 次没改善"，又要已跑够最小评估次数。
        if self.no_improve_count < self.exit_n:
            return False
        if self.eval_count < self.min_evals_before_stop:
            # 只在这一刻提示一次，避免每次评估都刷屏
            if self.no_improve_count == self.exit_n:
                logger.info(
                    f"连续 {self.no_improve_count} 次 val_loss 未刷新最优，但评估次数 "
                    f"{self.eval_count} 未达门槛 {self.min_evals_before_stop}，继续训练")
            return False

        logger.info(
            f"连续 {self.no_improve_count} 次 val_loss 未刷新最优({self.best_val_loss:.4f})，"
            f"已达 exit_n={self.exit_n}，建议停止训练")
        return True

    def best_dir(self) -> Optional[Path]:
        """当前最优权重所在目录；一份都没存过时返回 None。

        拿它去加载推理：
            from safetensors.torch import load_model
            sd = load_model(saver.best_dir() / "model.safetensors")
        """
        if not self.entries:
            return None
        return self.root_dir / self.entries[0][1]

    # -----------------------------------------------------------------------
    # 供 accelerator.register_for_checkpointing 使用
    # -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """返回**普通 Python 对象**（list of tuple / float / int）。

        accelerate 的 save_custom_state 会 torch.save(obj.state_dict(), ...)，
        所以这里不能返回 dataclass 或自定义类实例，保持朴素结构便于人工排查存档。
        """
        return {
            "entries": [(float(v), str(d)) for v, d in self.entries],
            "best_val_loss": float(self.best_val_loss),
            "no_improve_count": int(self.no_improve_count),
            "eval_count": int(self.eval_count),
        }

    def load_state_dict(self, state_dict) -> None:
        """load_state 时被 accelerate 自动调用，与 state_dict 成对。"""
        entries = [(float(v), str(d)) for v, d in state_dict.get("entries", [])]
        # 权重目录可能已被手动删掉，只保留确实存在的
        entries = [(v, d) for v, d in entries if (self.root_dir / d).is_dir()]
        entries.sort(key=lambda x: x[0])
        self.entries = entries[:self.topk]
        if self.entries:
            self.best_val_loss = self.entries[0][0]
        # 只有真正的断点续训才恢复早停计数 —— topk_meta.json 那条路不走这里（见 _restore）
        self.no_improve_count = int(state_dict.get("no_improve_count", 0))
        self.eval_count = int(state_dict.get("eval_count", 0))
        logger.info(
            f"从 accelerate 存档恢复 topk 记录 {len(self.entries)} 条，"
            f"历史最优 val_loss={self.best_val_loss:.4f}，早停计数={self.no_improve_count}")

    # -----------------------------------------------------------------------
    # 内部实现
    # -----------------------------------------------------------------------

    def _make_dir_name(self, val_loss, epoch_n, batch_n, seen_token_n) -> str:
        """目录名自带 loss/进度，人扫一眼就知道这份权重好不好。

        双下划线分隔字段名与值。rank 是**保存时的排名**（rank_1 即当时最优），
        淘汰后编号会重复、也不再代表当前名次 —— 真实排名看 topk_meta.json 的顺序。
        """
        rank = len(self.entries) + 1
        return (f"rank_{rank}__vl{float(val_loss):.4f}"
                f"__ep{epoch_n}__bt{batch_n}__tk{seen_token_n}")

    def _save_and_rank(self, model, train_loss, val_loss, epoch_n, batch_n, seen_token_n) -> None:
        dir_name = self._make_dir_name(val_loss, epoch_n, batch_n, seen_token_n)
        save_dir = self.root_dir / dir_name
        # unwrap_model：单卡下返回原对象（零成本），多卡下剥掉 DDP 包装，
        # 否则 state_dict 的键会带 module. 前缀，存下来的权重裸 KpcLLMModel 加载不了。
        # save_model 默认 safe_serialization=True，产出 model.safetensors。
        self.accelerator.save_model(self.accelerator.unwrap_model(model), str(save_dir))

        self.entries.append((float(val_loss), dir_name))
        self.entries.sort(key=lambda x: x[0])
        while len(self.entries) > self.topk:      # 淘汰跌出 topk 的
            self._drop(self.entries.pop()[1])

        self._save_meta()
        logger.info(
            f"保存 val_loss={float(val_loss):.4f} 的权重到 {dir_name}；"
            f"当前 top{self.topk}: " + ", ".join(f"{v:.4f}" for v, _ in self.entries))

    def _drop(self, dir_name: str) -> None:
        """删掉一份权重目录。

        ignore_errors=True：Windows 下目录正被其他进程打开时删不掉（权重可能正被
        一个推理进程加载着），但淘汰只是 housekeeping，不该让数小时的训练崩在这里。
        """
        shutil.rmtree(self.root_dir / dir_name, ignore_errors=True)
        logger.info(f"淘汰跌出 top{self.topk} 的权重: {dir_name}")

    def _save_meta(self) -> None:
        """把当前 topk 记录写进 topk_meta.json，让 topk 目录自包含。"""
        self.meta_path.write_text(json.dumps({
            "entries": [[v, d] for v, d in self.entries],
            "best_val_loss": self.best_val_loss,
            "no_improve_count": self.no_improve_count,
            "eval_count": self.eval_count,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_meta(self) -> Optional[dict]:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"topk_meta.json 损坏，本次从空开始: {exc}")
            return None

    def _restore(self) -> None:
        """构造时从 topk_meta.json 恢复 topk 列表。

        这里**不恢复** no_improve_count：这份 json 可能来自几天前的一次运行，
        拿旧的早停计数接着等会误杀本该继续跑的实验。真正需要精确恢复早停计数的
        场景（训练中断后接着练）走 load_state -> load_state_dict，那条路才恢复。
        """
        meta = self._load_meta()
        if not meta:
            return
        entries = [(float(v), str(d)) for v, d in meta.get("entries", [])]
        entries = [(v, d) for v, d in entries if (self.root_dir / d).is_dir()]
        entries.sort(key=lambda x: x[0])
        self.entries = entries[:self.topk]
        if self.entries:
            self.best_val_loss = self.entries[0][0]
            logger.info(
                f"恢复已有 topk 权重 {len(self.entries)} 份，历史最优 val_loss={self.best_val_loss:.4f}")
