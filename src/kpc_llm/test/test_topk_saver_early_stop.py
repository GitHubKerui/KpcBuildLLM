"""TopKSaver 早停逻辑测试：min_delta 与 min_evals_before_stop。

用替身 accelerator 避免真实落盘，专注验证"什么时候该早停"这一判定。
"""

import pytest

from kpc_llm.kpc_llm_model.pretraining.topk_saver import TopKSaver


class _FakeAccelerator:
    """只实现 TopKSaver 用到的接口，避免引入 Accelerate / 真实写盘。"""

    def register_for_checkpointing(self, obj):
        pass

    def unwrap_model(self, model):
        return model

    def save_model(self, model, save_dir):
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)


def _saver(tmp_path, **kwargs) -> TopKSaver:
    kwargs.setdefault("topk", 3)
    kwargs.setdefault("exit_n", 3)
    return TopKSaver(_FakeAccelerator(), root_dir=str(tmp_path), **kwargs)


def _feed(saver: TopKSaver, losses):
    """依次把 val_loss 喂给 saver，返回每次 step() 的返回值。"""
    return [saver.step(None, 0.0, v, 0, i, 0) for i, v in enumerate(losses)]


# ---------------------------------------------------------------------------
# min_delta：改善不到阈值就当作噪声，不重置早停计数
# ---------------------------------------------------------------------------
def test_min_delta_treats_small_improvements_as_noise(tmp_path):
    saver = _saver(tmp_path, exit_n=3, min_delta=0.01)
    # 首次 3.0 建立基线；之后三次"改善"都 < 0.01，应被当作没改善累加计数
    assert _feed(saver, [3.0, 2.995, 2.994, 2.993]) == [False, False, False, True]
    assert saver.no_improve_count == 3
    assert saver.best_val_loss == pytest.approx(3.0)


def test_real_improvement_beyond_min_delta_resets_counter(tmp_path):
    saver = _saver(tmp_path, exit_n=3, min_delta=0.01)
    # 2.98 相对 3.0 改善 0.02 > 0.01 -> 重置计数；之后两次小改善不重置，但未达 exit_n=3
    assert _feed(saver, [3.0, 2.98, 2.975, 2.974]) == [False, False, False, False]
    assert saver.no_improve_count == 2
    assert saver.best_val_loss == pytest.approx(2.98)


def test_min_delta_zero_never_stops_when_always_improving(tmp_path):
    saver = _saver(tmp_path, exit_n=3, min_delta=0.0)
    assert _feed(saver, [3.0, 2.9, 2.8, 2.7, 2.6]) == [False] * 5


# ---------------------------------------------------------------------------
# min_evals_before_stop：评估次数不到门槛，即使满足 exit_n 也不早停
# ---------------------------------------------------------------------------
def test_min_evals_before_stop_withholds_early_stop(tmp_path):
    saver = _saver(tmp_path, exit_n=1, min_delta=0.0, min_evals_before_stop=5)
    # 第 2 次起就满足 exit_n=1，但评估次数不到 5 一律不早停
    assert _feed(saver, [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]) == \
        [False, False, False, False, True, True]
    assert saver.eval_count == 6


def test_no_gate_stops_as_soon_as_patience_exhausted(tmp_path):
    saver = _saver(tmp_path, exit_n=1, min_delta=0.0, min_evals_before_stop=0)
    assert _feed(saver, [3.0, 3.1]) == [False, True]


# ---------------------------------------------------------------------------
# 参数校验 & 状态持久化
# ---------------------------------------------------------------------------
def test_negative_params_rejected(tmp_path):
    with pytest.raises(ValueError):
        _saver(tmp_path, min_delta=-0.1)
    with pytest.raises(ValueError):
        _saver(tmp_path, min_evals_before_stop=-1)


def test_eval_count_survives_state_dict_roundtrip(tmp_path):
    saver = _saver(tmp_path, exit_n=5, min_delta=0.0)
    _feed(saver, [3.0, 3.1, 3.2])
    state = saver.state_dict()
    assert state["eval_count"] == 3

    restored = _saver(tmp_path, exit_n=5, min_delta=0.0)
    restored.load_state_dict(state)
    assert restored.eval_count == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
