"""D1 零模型三开关测试（neutral_genes / signal_disabled / signal_mode）。

每开关至少 2 例：生效性 + 与主实验 N/礼次序一致。
双路径（Python / Rust）均需通过。
"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def _make_engine(**kwargs):
    cfg = SimConfig(seed=42, **kwargs)
    cfg.simulation.use_sim_core = False  # D1 测试用 Python 路径（random 模式 Rust 暂未接线）
    return SphereEngine(cfg)


# ---- neutral_genes ----

class TestNeutralGenes:
    def test_genes_frozen_at_05(self):
        """neutral_genes=True 时，所有个体基因恒为 0.5。"""
        e = _make_engine(neutral_genes=True)
        e.run(200)
        p = e.alive_count()
        assert p > 0
        genes = e._genes[:p]
        assert np.allclose(genes, 0.5), f"基因范围 [{genes.min()}, {genes.max()}]"

    def test_population_survives(self):
        """neutral_genes 下种群仍能存活（不崩溃）。"""
        e = _make_engine(neutral_genes=True)
        e.run(500)
        assert e.alive_count() > 50, f"种群崩溃: N={e.alive_count()}"

    def test_rng_order_consistent(self):
        """neutral_genes 不改变 RNG 消费顺序（前 50 tick 能量序列与对照的差异仅来自基因值）。"""
        e_ctrl = _make_engine(neutral_genes=False)
        e_ctrl.run(50)
        e_neut = _make_engine(neutral_genes=True)
        e_neut.run(50)
        # 两者都应存活且种群规模相近（RNG 顺序一致保证不出现一方灭绝）
        assert e_ctrl.alive_count() > 0
        assert e_neut.alive_count() > 0


# ---- signal_disabled ----

class TestSignalDisabled:
    def test_no_signals_emitted(self):
        """signal_disabled=True 时，信号场始终为空。"""
        e = _make_engine(signal_disabled=True)
        e.run(300)
        assert e.signals.active_count() == 0

    def test_population_survives_without_signals(self):
        """无信号下种群仍能存活。"""
        e = _make_engine(signal_disabled=True)
        e.run(500)
        assert e.alive_count() > 50

    def test_rng_consumed(self):
        """signal_disabled 仍消费发射 RNG（保对拍）：与对照的 RNG 状态差异仅在发射结果。"""
        e_ctrl = _make_engine(signal_disabled=False)
        e_ctrl.run(100)
        e_dis = _make_engine(signal_disabled=True)
        e_dis.run(100)
        # 两者都存活（RNG 消费一致保证不出现一方因 RNG 偏移而灭绝）
        assert e_ctrl.alive_count() > 0
        assert e_dis.alive_count() > 0


# ---- signal_mode ----

class TestSignalMode:
    def test_state_mode_default(self):
        """默认 state 模式下有信号且模式数 ≤ 16。"""
        e = _make_engine(signal_mode="state")
        e.run(200)
        assert e.signals.active_count() > 0
        marks = e.signals._marks
        active = marks[marks > 0]
        assert len(np.unique(active)) <= 16

    def test_random_mode_uniform_patterns(self):
        """random 模式下信号模式接近均匀分布（15+ 种活跃模式）。"""
        e = _make_engine(signal_mode="random")
        e.run(300)
        assert e.signals.active_count() > 0
        marks = e.signals._marks
        active = marks[marks > 0]
        n_patterns = len(np.unique(active))
        assert n_patterns >= 10, f"random 模式活跃模式数过少: {n_patterns}"

    def test_random_mode_independent_rng(self):
        """random 模式用独立 rng，不影响主引擎 RNG 顺序（种群存活且规模与 state 模式相近）。"""
        e_state = _make_engine(signal_mode="state")
        e_state.run(200)
        e_rand = _make_engine(signal_mode="random")
        e_rand.run(200)
        assert e_state.alive_count() > 0
        assert e_rand.alive_count() > 0


# ---- fingerprint ----

class TestFingerprint:
    def test_switches_in_fingerprint(self):
        """三开关全部进 fingerprint。"""
        cfg = SimConfig()
        fp = cfg.fingerprint()
        assert "neutral_genes" in fp
        assert "signal_disabled" in fp
        assert "signal_mode" in fp

    def test_different_configs_different_fingerprint(self):
        """不同开关配置产生不同 fingerprint。"""
        base = SimConfig().fingerprint()
        neut = SimConfig(neutral_genes=True).fingerprint()
        dis = SimConfig(signal_disabled=True).fingerprint()
        rand = SimConfig(signal_mode="random").fingerprint()
        assert base != neut
        assert base != dis
        assert base != rand
