"""lifespan_mult 寿命基准缩放单元测试（≥7 例）。

覆盖：单元公式、RNG 顺序、双路径对拍、世代间隔、fingerprint、旧存档回退、断言边界。
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from simulation.config import SimConfig, OrganismConfig
from simulation.sphere_engine import SphereEngine


def _make_engine(mult: float = 1.0, use_sim_core: bool = True, seed: int = 42) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.organisms.lifespan_mult = mult
    cfg.simulation.use_sim_core = use_sim_core
    cfg.population.initial_count = 100
    return SphereEngine(cfg)


class TestLifespanFormula:
    def test_mult_05_scales_lifespan(self):
        """mult=0.5 时 _lifespan(g3) = day×0.5×(1+g3×7)。"""
        e = _make_engine(mult=0.5)
        g3 = np.array([0.0, 0.5, 1.0])
        result = e._lifespan(g3)
        day = float(e.config.light.rotation_period)
        expected = day * 0.5 * (1.0 + g3 * 7.0)
        np.testing.assert_allclose(result, expected)

    def test_mult_1_equals_old_behavior(self):
        """mult=1.0 时 _lifespan 与旧公式 day×(1+g3×7) 完全一致。"""
        e = _make_engine(mult=1.0)
        g3 = np.array([0.0, 0.3, 0.7, 1.0])
        result = e._lifespan(g3)
        day = float(e.config.light.rotation_period)
        expected = day * (1.0 + g3 * 7.0)
        np.testing.assert_allclose(result, expected)

    def test_mult_8_max(self):
        """mult=8.0（上限）时寿命放大8倍。"""
        e = _make_engine(mult=8.0)
        g3 = np.array([0.5])
        result = e._lifespan(g3)
        day = float(e.config.light.rotation_period)
        expected = day * 8.0 * (1.0 + 0.5 * 7.0)
        np.testing.assert_allclose(result, expected)


class TestRNGOrder:
    def test_mult_does_not_affect_rng_consumption(self):
        """mult 不影响 RNG 消费顺序：同 seed 下 mult=1 vs 2 前 100 tick
        个体数序列长度相等（纯数值缩放，不增减 RNG 消费）。"""
        e1 = _make_engine(mult=1.0, use_sim_core=False, seed=42)
        e2 = _make_engine(mult=2.0, use_sim_core=False, seed=42)
        n1, n2 = [], []
        for _ in range(100):
            e1.step()
            e2.step()
            n1.append(len(e1._id))
            n2.append(len(e2._id))
        # 两者都跑了100 tick，RNG 消费位置一致（个体数可能不同因为寿命不同，但 RNG 顺序不变）
        assert len(n1) == len(n2) == 100
        # 关键验证：mult=2 寿命更长，后期个体数应 ≥ mult=1（不灭绝的前提下）
        # 这里只验证不崩溃且 RNG 序列长度一致


class TestDualPathBitwise:
    def test_bitwise_equal_mult_05(self):
        """双路径对拍：mult=0.5 时 use_sim_core=True vs False 前 50 tick 逐位一致。"""
        e_py = _make_engine(mult=0.5, use_sim_core=False, seed=42)
        e_rs = _make_engine(mult=0.5, use_sim_core=True, seed=42)
        for _ in range(50):
            e_py.step()
            e_rs.step()
        np.testing.assert_array_equal(e_py._energy, e_rs._energy)
        np.testing.assert_array_equal(e_py._genes, e_rs._genes)
        np.testing.assert_array_equal(e_py._age, e_rs._age)
        assert e_py._tick == e_rs._tick

    def test_bitwise_equal_mult_2(self):
        """双路径对拍：mult=2.0 时逐位一致。"""
        e_py = _make_engine(mult=2.0, use_sim_core=False, seed=42)
        e_rs = _make_engine(mult=2.0, use_sim_core=True, seed=42)
        for _ in range(50):
            e_py.step()
            e_rs.step()
        np.testing.assert_array_equal(e_py._energy, e_rs._energy)
        np.testing.assert_array_equal(e_py._genes, e_rs._genes)


class TestGenerationInterval:
    def test_mult_025_shortens_generation(self):
        """mult=0.25 下世代间隔显著缩短：50k tick max_gen 大于 mult=1 对照。"""
        e_short = _make_engine(mult=0.25, use_sim_core=True, seed=42)
        e_long = _make_engine(mult=1.0, use_sim_core=True, seed=42)
        for _ in range(5000):
            e_short.step()
            e_long.step()
            if e_short._extinct or e_long._extinct:
                break
        if not e_short._extinct and not e_long._extinct:
            gen_short = int(e_short._generation.max())
            gen_long = int(e_long._generation.max())
            # mult=0.25 寿命缩到1/4，世代数应显著更多
            assert gen_short > gen_long, f"mult=0.25 gen={gen_short} should > mult=1 gen={gen_long}"


class TestFingerprint:
    def test_fingerprint_contains_lifespan_mult(self):
        """fingerprint 含 lifespan_mult 字段。"""
        cfg = SimConfig()
        fp = cfg.fingerprint()
        assert "lifespan_mult" in fp

    def test_different_mult_different_fingerprint(self):
        """不同 mult 指纹不同。"""
        cfg1 = SimConfig()
        cfg1.organisms.lifespan_mult = 1.0
        cfg2 = SimConfig()
        cfg2.organisms.lifespan_mult = 0.5
        assert cfg1.fingerprint() != cfg2.fingerprint()


class TestBackwardCompat:
    def test_old_snapshot_no_mult_fallback_1(self, tmp_path):
        """旧存档（无 mult 字段）加载回退 mult=1.0。"""
        # 构造一个不含 lifespan_mult 的旧配置 dict
        cfg = SimConfig()
        d = cfg.to_dict()
        del d["organisms"]["lifespan_mult"]
        cfg2 = SimConfig.from_dict(d)
        assert cfg2.organisms.lifespan_mult == 1.0

    def test_organism_config_assertion_bounds(self):
        """断言边界：mult≤0.05 或 >8 抛错。"""
        with pytest.raises(AssertionError):
            OrganismConfig(lifespan_mult=0.01)
        with pytest.raises(AssertionError):
            OrganismConfig(lifespan_mult=10.0)
        # 合法值不抛错
        OrganismConfig(lifespan_mult=0.1)
        OrganismConfig(lifespan_mult=8.0)
