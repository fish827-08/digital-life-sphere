"""引擎生命周期机制的回归测试（模块二）。

覆盖：成熟年龄门槛、随年龄变化的维持需求、基因扩充（g4~g13）已接入、
同配置同种子的确定性、长跑不崩。
"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine

EQUATOR = 60 * 60 + 30  # 赤道某格


def make_engine(n=1, seed=3):
    cfg = SimConfig(seed=seed)
    cfg.population.initial_count = n
    cfg.simulation.ticks = 10_000
    return SphereEngine(cfg)


def set_genes(engine, row, **kw):
    """把单个体的基因链设成给定值（未给定位默认 0.5）。kw 键如 'g0'='g13'。"""
    g = np.zeros((1, 16), dtype=np.float64)
    g[0, :] = 0.5
    for k, v in kw.items():
        g[0, int(k.lstrip("g"))] = v
    engine._genes = g


def first_birth_tick(engine, run_ticks):
    for t in range(1, run_ticks + 1):
        if engine.step().born:
            return t
    return None


# ---------------------------------------------------------------------------
# 成熟年龄门槛：未成熟一律不能繁衍
# ---------------------------------------------------------------------------

def test_maturity_gate_blocks_juvenile_reproduction():
    """未成熟（年龄 < 寿命×15%）绝不能繁衍。
    g3=0.0 → 寿命 = 一昼夜(2400)×1 = 2400 → 成熟年龄 360 tick。"""
    e = make_engine()
    set_genes(e, 0, g0=0.0, g2=0.0, g3=0.0, g5=1.0)
    e._flat[:] = EQUATOR
    e._energy[0] = 300.0
    e.resources._grid[:] = 50.0
    birth = first_birth_tick(e, 700)
    assert birth is not None
    assert birth >= 360, f"未成熟不应繁衍，却在第 {birth} tick 生"


def test_disable_maturity_reproduces_immediately():
    """把成熟比例调到几乎 0 → 一出生立刻能生（作对照，证明是门槛在起作用）。"""
    e = make_engine()
    set_genes(e, 0, g0=0.0, g2=0.0, g5=1.0)
    e.config.organisms.maturity_fraction = 1e-6
    e._flat[:] = EQUATOR
    e._energy[0] = 300.0
    e.resources._grid[:] = 50.0
    birth = first_birth_tick(e, 50)
    assert birth is not None and birth <= 10


# ---------------------------------------------------------------------------
# 随年龄变化的能量需求：幼体耗能 > 成年
# ---------------------------------------------------------------------------

def test_maintenance_depends_on_age():
    """同一只个体，幼体期每 tick 维持耗能高于成年期（在长身体）。"""

    def drain(age):
        e = make_engine()
        g = np.zeros((1, 16), dtype=np.float64)
        g[0, :] = 0.0
        g[0, 1] = 1.0   # metab_mult = 2.0（确定性能耗）
        g[0, 3] = 1.0   # 寿命 = 2400×8 = 19200（成熟2880 / 老年14400，两点都在区间内）
        g[0, 9] = 1.0   # 恒温，不受温度偏好干扰
        e._genes = g
        e._flat[:] = EQUATOR
        e._energy[0] = 50.0
        e._stomach[0] = 0.0
        e.resources._grid[:] = 0.0  # 无外界能量，纯看维持
        e._age[0] = age
        for _ in range(10):
            e.step()
        return float(e._energy[0])

    juv = drain(0)        # 幼体（0 < 成熟 2880）
    adult = drain(4000)   # 成年（2880 < 4000 < 老年 14400）
    assert juv < adult, "幼体维持耗能应高于成年"
    assert (50 - juv) - (50 - adult) > 3.0, "年龄能耗差太小"


# ---------------------------------------------------------------------------
# 基因扩充 g4~g13 已接入（抽查两个代表性的机制）
# ---------------------------------------------------------------------------

def test_photosynthesis_income_g8():
    """有光合基因（g8=1）的个体在白天比没有的（g8=0）多获得少量能量。"""
    def run(g8):
        e = make_engine()
        g = np.zeros((1, 16), dtype=np.float64)
        g[0, :] = 0.0
        g[0, 0] = 0.0   # 不动
        g[0, 1] = 0.0   # 代谢最低
        g[0, 3] = 1.0   # 寿命 = 2400×8 = 19200（成熟2880 / 老年14400）
        g[0, 8] = g8
        e._genes = g
        e._flat[:] = EQUATOR  # 赤道
        e._tick = 597         # 连续 5 tick 都落在赤道正午前后
        e._energy[0] = 200.0
        e.resources._grid[:] = 0.0
        e._age[0] = 4000      # 成年（2880 < 4000 < 14400）
        for _ in range(5):
            e.step()
        return float(e._energy[0])

    with_photo = run(1.0)
    without_photo = run(0.0)
    assert with_photo > without_photo, "光合应带来额外能量"
    assert with_photo - without_photo > 0.3, "光合产能收益过低"


def test_repro_cooldown_g12():
    """冷却期内攒再多能量也不再生。"""
    e = make_engine()
    set_genes(e, 0, g0=0.0, g2=0.5, g7=0.05, g12=1.0)
    e._flat[:] = EQUATOR
    e._energy[0] = 300.0
    e.resources._grid[:] = 50.0
    e._age[0] = 2000  # 已成熟（g3=0.5 → 寿命10800 → 成熟年龄1620）
    e._repro_cooldown[0] = 60.0  # 强制冷却
    births = sum(e.step().born for _ in range(5))
    assert births == 0, "冷却未过却繁衍了"


# ---------------------------------------------------------------------------
# 确定性 & 长跑稳定性
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """同配置 + 同种子 → 生成的历史完全一致。"""
    a = SphereEngine(SimConfig(seed=7))
    b = SphereEngine(SimConfig(seed=7))
    ha = a.run(200)
    hb = b.run(200)
    for ta, tb in zip(ha, hb):
        assert ta.to_dict() == tb.to_dict()


def test_long_run_stability():
    """1000 tick 长跑不崩、不灭绝、状态数组长度一致。"""
    e = SphereEngine(SimConfig(seed=123))
    h = e.run(1000)
    assert not e.extinct and e.alive_count() > 0
    assert len(e._id) == len(e._genes) == len(e._repro_cooldown)
    assert len(h) == 1000