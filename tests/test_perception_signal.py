"""L3 感知(g14)与信号(g15)基因单元测试。"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def make_engine(n=10, seed=42):
    cfg = SimConfig(seed=seed)
    cfg.population.initial_count = n
    cfg.simulation.ticks = 1000
    return SphereEngine(cfg)


def test_engine_has_signal_field():
    """引擎创建后有 signals 属性（SignalField 实例）。"""
    e = make_engine(n=5)
    assert hasattr(e, "signals")
    assert e.signals.active_count() == 0


def test_signal_field_ticks_with_engine():
    """引擎 step() 后信号场时间推进（标记衰减）。"""
    e = make_engine(n=1)
    e.signals.write(100, 0b1010)
    assert e.signals.read(100) == 0b1010
    for _ in range(e.signals.duration):
        e.step()
    assert e.signals.read(100) == 0


def test_g15_high_emits_signals():
    """高 g15 个体更可能发射信号（多次 step 后场上有标记）。"""
    e = make_engine(n=20, seed=7)
    e._genes[:, 15] = 1.0  # 全部高发射倾向
    e._energy[:] = e.config.organisms.max_energy  # 付得起发射成本
    for _ in range(20):
        e.step()
    assert e.signals.active_count() > 0, "高 g15 应该产生信号标记"


def test_g15_zero_no_signals():
    """g15=0 的个体不发射信号。"""
    e = make_engine(n=20, seed=7)
    e._genes[:, 15] = 0.0
    for _ in range(20):
        e.step()
    assert e.signals.active_count() == 0


def test_signal_emit_costs_energy():
    """发射信号消耗能量（0.5/次）。"""
    e = make_engine(n=1, seed=3)
    e._genes[0, 15] = 1.0
    e._genes[0, 0] = 0.0  # 不移动，避免移动耗能干扰
    e._flat[0] = 500
    e._energy[0] = 10.0
    e.resources._grid[:] = 0.0  # 无食物，避免进食干扰
    # 手动触发一次发射（g15=1，rng.random < 1 必发射）
    e_before = e._energy[0]
    e.step()
    # 能量应该减少（至少包含发射成本 0.5 + 维持消耗）
    assert e._energy[0] < e_before


def test_signal_pattern_in_range():
    """信号模式在 0~15 范围内（4 位田字格）。"""
    e = make_engine(n=30, seed=11)
    e._genes[:, 15] = 1.0
    e._energy[:] = e.config.organisms.max_energy
    for _ in range(10):
        e.step()
    marks = e.signals._marks
    assert (marks >= 0).all() and (marks <= 15).all()


def test_g14_perception_moves_to_food():
    """高 g14 个体倾向移向食物多的邻格。

    构造：个体在中间格，左邻食物满、右邻食物空，高 g14 → 应选左邻。
    """
    e = make_engine(n=1, seed=5)
    e._genes[0, 0] = 1.0  # 必移动
    e._genes[0, 14] = 1.0  # 高感知
    e._genes[0, 13] = 0.5  # 中性群居（不干扰）
    e._energy[0] = e.config.organisms.max_energy
    # 选一个有 2+ 邻居的格子（赤道非极点）
    center = 60 * 30 + 60  # 赤道某格
    e._flat[0] = center
    nb = e.world.neighbors(center)
    # 把第一个邻格食物填满，其余清空
    e.resources._grid[:] = 0.0
    e.resources._capacity[:] = 100.0
    e.resources._grid[nb[0]] = 100.0
    e.step()
    # 应该移到食物多的邻格（nb[0]）
    assert e._flat[0] == nb[0], f"高 g14 应移向食物格，实际移到 {e._flat[0]}"


def test_g14_zero_random_move():
    """g14=0 且 g13=0.5 的个体移动随机（不偏向食物或邻居）。"""
    e = make_engine(n=1, seed=5)
    e._genes[0, 0] = 1.0
    e._genes[0, 14] = 0.0
    e._genes[0, 13] = 0.5
    e._energy[0] = e.config.organisms.max_energy
    center = 60 * 30 + 60
    e._flat[0] = center
    nb = e.world.neighbors(center)
    e.resources._grid[:] = 0.0
    e.resources._capacity[:] = 100.0
    e.resources._grid[nb[0]] = 100.0
    # 多次移动，不应总是选 nb[0]（随机）
    choices = set()
    for _ in range(20):
        e._flat[0] = center
        e.step()
        choices.add(int(e._flat[0]))
    assert len(choices) > 1, "g14=0/g13=0.5 应随机移动，不应固定一格"


def test_info_gain_from_signals():
    """所在格有信号时，愉悦度信息增益为正（valence 更高）。

    构造两个相同个体，一个在有信号格，一个在无信号格，
    比较愉悦度更新后的 valence。
    """
    e = make_engine(n=2, seed=9)
    e._flat[:] = [100, 200]
    e._energy[:] = e.config.organisms.max_energy * 0.5
    e.resources._grid[:] = e.resources._capacity[:] * 0.5
    e._expectation[:, :] = 0.5  # 统一预期
    # 在 100 格放信号，200 格不放
    e.signals.write(100, 0b1111)
    energy_before = e._energy.copy()
    e._update_pleasure(2, energy_before)
    # 有信号的个体 valence 应高于无信号的
    assert e._valence[0] > e._valence[1], (
        f"有信号个体 valence={e._valence[0]:.3f} 应高于无信号 {e._valence[1]:.3f}"
    )


def test_perception_and_signal_long_run():
    """g14+g15 系统长跑 300 tick 不崩、数组形状正确。"""
    e = make_engine(n=50, seed=123)
    for _ in range(300):
        e.step()
        if e.extinct:
            break
    n = len(e._id)
    assert e._valence.shape == (n,)
    assert e._expectation.shape == (n, e.config.pleasure.expectation_size)
    # 信号场标记在合法范围
    assert (e.signals._marks >= 0).all() and (e.signals._marks <= 15).all()
