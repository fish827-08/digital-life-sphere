"""D-8：V-1 oracle 正向对照（R39，利益对齐型 / Lewis 共利）。

对应规格 `_share/规格-V1-oracle引擎级-20260913.md`（v1.2）：
C-2 守恒 / C-3 不为负 / C-9 保本封顶 / O-4 / O-6 / O-7 / 快照双向兼容（C-7）。
实现层两处裁定（已在讨论板登记）：归因键=_id；窗口按"剩余寿命"语义
（`age >= duration - persistence`，规格 §2.3 原式方向反了）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.oracle import EMISSION_COST, apply_oracle, attribution_ok  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


# ---------------- 纯转移核（函数级，C-2/C-3/C-9） ---------------------------

def test_apply_oracle_conserves_energy():
    energy = np.array([10.0, 5.0, 1.0, 100.0])
    before = energy.sum()
    total, cnt, _, _ = apply_oracle(
        energy=energy,
        receiver_slots=np.array([1, 2], dtype=np.int64),
        sender_slots=np.array([0, 0], dtype=np.int64),
        budget=np.array([1.0, 1.0], dtype=np.float64),
        donation=0.05,
    )
    assert cnt == 2
    assert abs(total - 0.1) < 1e-12
    assert abs(energy.sum() - before) < 1e-9          # O-4 守恒
    assert (energy >= 0).all()                         # C-3 不为负


def test_apply_oracle_respects_budget_and_energy():
    """预算耗尽 ⇒ 不转移；发送者买不起 ⇒ 只转得起的部分。"""
    energy = np.array([0.02, 5.0])
    total, cnt, _, _ = apply_oracle(
        energy=energy,
        receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([1.0], dtype=np.float64),
        donation=0.05,
    )
    assert cnt == 1 and abs(total - 0.02) < 1e-12      # 被能量卡住
    assert abs(energy[0]) < 1e-12

    energy2 = np.array([10.0, 5.0])
    total2, cnt2, _, _ = apply_oracle(
        energy=energy2,
        receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([0.01], dtype=np.float64),     # 被预算卡住（C-9）
        donation=0.05,
    )
    assert cnt2 == 1 and abs(total2 - 0.01) < 1e-12

    total3, cnt3, _, _ = apply_oracle(
        energy=energy2,
        receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([0.0], dtype=np.float64),
        donation=0.05,
    )
    assert cnt3 == 0 and total3 == 0.0


def test_attribution_window_remaining_lifetime_semantics():
    """SignalField._age 是剩余寿命（写入=duration，每 tick −1）。

    "最近 persistence tick 内写入" ⇔ age >= duration − persistence。
    规格 §2.3 的 `age <= persistence` 按此语义方向相反（已在讨论板登记更正）。
    """
    age = np.array([50, 45, 41, 40, 39, 20, 0], dtype=np.int32)
    ok = attribution_ok(age, signal_duration=50, persistence=10)
    assert ok.tolist() == [True, True, True, True, False, False, False]
    ok0 = attribution_ok(age, signal_duration=50, persistence=0)   # 仅本 tick
    assert ok0.tolist() == [True, False, False, False, False, False, False]


# ---------------- 引擎级（O-6 / O-7 / C-7 / C-8） ---------------------------

def _engine(oracle: bool, seed: int = 42) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    d2 = InfoStructureConfig(enabled=True, learning_rate=0.05)
    cfg.info_structure = d2
    cfg.oracle.enabled = oracle
    return SphereEngine(cfg)


def _run_recording(e: SphereEngine, ticks: int) -> list[int]:
    draws = []
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
        draws.append(e.rng_draws)
    return draws


def test_o6_same_tick_rng_draws_until_first_transfer():
    """O-6：oracle 开/关两臂，在首次转移发生前每 tick 的 rng_draws 逐位相同
    （零新增 RNG 调用点 C-1 的可执行化）。首次转移所在 tick 起允许分岔（C-1′ 实验效应）。

    ⚠️ F-D2：全局 np.random 进程级共享 ⇒ 先跑完 a 再构造 b（见 test_d18 同款注释）。
    """
    a = _engine(oracle=True)
    da: list[int] = []
    fired_at = None
    for i in range(400):
        if a.extinct:
            break
        a.step()
        da.append(a.rng_draws)
        if fired_at is None and a.oracle_stats()["count"] > 0:
            fired_at = i + 1
    b = _engine(oracle=False)          # 重新 seed 全局 np.random ⇒ 与 a 起点一致
    db: list[int] = []
    for _ in range(len(da)):
        if b.extinct:
            break
        b.step()
        db.append(b.rng_draws)
    if fired_at is None:
        assert da == db, "oracle 未触发时随机流必须完全一致"
        return
    assert da[: fired_at - 1] == db[: fired_at - 1], (
        "首次转移前每 tick 的 RNG 消费必须逐位相同（C-1）"
    )


def test_o7_return_ratio_bounded_by_one():
    """O-7 / C-9：保本封顶 ⇒ return_ratio ≤ 1（由构造保证，实测复核）。"""
    e = _engine(oracle=True)
    for _ in range(400):
        if e.extinct:
            break
        e.step()
    s = e.oracle_stats()
    assert s["enabled"] is True
    assert s["oracle_return_ratio"] <= 1.0 + 1e-9
    assert s["transfers"] >= 0.0


def test_c7_snapshot_roundtrip_keeps_oracle_state():
    """C-7：_last_sender / 保本记账必须进快照；续跑后 oracle 统计与连续跑一致。

    ⚠️ F-D2：引擎快照不含全局 np.random 状态（已知缺陷）⇒ 测试按生产侧
    （a4_verify_capacity 的 rngstate.pkl）同款模式补存取，否则续跑必然分岔。
    """
    import pickle

    a = _engine(oracle=True)
    for _ in range(60):
        a.step()
        if a.extinct:
            break
    path = Path(__file__).parent / "_tmp_oracle_snap.npz"
    rng_state_path = path.with_suffix(".rngstate.pkl")
    a.save_snapshot(str(path))
    with open(rng_state_path, "wb") as fh:
        pickle.dump(np.random.get_state(), fh)          # F-D2 补偿
    np_state_before_load = np.random.get_state()
    b = SphereEngine.load_snapshot(str(path))
    with open(rng_state_path, "rb") as fh:
        np.random.set_state(pickle.load(fh))            # 恢复全局态
    del np_state_before_load
    assert np.array_equal(b._last_sender, a._last_sender)
    assert np.array_equal(b._emit_count, a._emit_count)
    assert np.allclose(b._oracle_gain, a._oracle_gain)
    assert b.oracle_stats()["count"] == a.oracle_stats()["count"]
    # 续跑各 40 tick：oracle 记账仍一致。
    # ⚠️ 全局 np.random 是进程级单例 ⇒ 两条续跑【不能交错 step】，须各自
    # 从分叉点状态独立重放（先 a 后 b，每次续跑前恢复分叉点全局态）。
    with open(rng_state_path, "rb") as fh:
        np.random.set_state(pickle.load(fh))
    for _ in range(40):
        if a.extinct:
            break
        a.step()
    a_count = a.oracle_stats()["count"]
    with open(rng_state_path, "rb") as fh:
        np.random.set_state(pickle.load(fh))
    for _ in range(40):
        if b.extinct:
            break
        b.step()
    assert b.oracle_stats()["count"] == a_count
    path.unlink(missing_ok=True)
    rng_state_path.unlink(missing_ok=True)


def test_c8_sim_core_plus_oracle_raises():
    """C-8：use_sim_core=True + oracle.enabled=True ⇒ 引擎初始化显式报错。"""
    try:
        import sim_core  # noqa: F401
        has_core = True
    except ImportError:
        has_core = False
    if not has_core:
        pytest.skip("sim_core 未安装（本机无 Rust 扩展）——C-8 在导入处已报错")
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = True
    cfg.oracle.enabled = True
    with pytest.raises(RuntimeError, match="C-8"):
        SphereEngine(cfg)


def test_oracle_config_dict_fallback():
    """C-7：旧存档缺 oracle 键 ⇒ 回退默认关闭。"""
    d = SimConfig(seed=1).to_dict()
    d.pop("oracle", None)
    c2 = SimConfig.from_dict(d)
    assert c2.oracle.enabled is False
    assert c2.oracle.donation == 0.05
    # fingerprint 含 oracle（批次自证是否开了 oracle —— 规格 §七.4）
    assert "oracle" in SimConfig(seed=1).fingerprint()
    assert EMISSION_COST == 0.1
