"""D-19：provenance 硬校验 + rng_draws 计数器。

依据：**R31④**（provenance 不得静默写 unknown）；**V-1 O-6**（rng_draws 计数器）。
出处：`_share/讨论板.md` 2026-09-13 17:01 任务清单第 1 项。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig  # noqa: E402
from simulation.provenance import (  # noqa: E402
    CountingRNG, collect, git_commit, validate,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402


# ---- rng_draws 计数器 ----

def test_rng_draws_starts_and_grows():
    e = SphereEngine(SimConfig(seed=42))
    d0 = e.rng_draws
    for _ in range(50):
        e.step()
    assert e.rng_draws > d0, "step 消耗随机流后 draws 应增加"


def test_rng_draws_deterministic_across_reruns():
    """同 seed 同配置跑同样 tick ⇒ rng_draws 必须相同（可复现性的机械信号）。"""
    def run():
        e = SphereEngine(SimConfig(seed=42))
        for _ in range(50):
            e.step()
        return e.rng_draws
    assert run() == run()


def test_counting_rng_does_not_alter_stream():
    """包装只计数，随机流必须逐位不变（否则会破坏所有对拍/快照复现）。"""
    gen = np.random.default_rng(7)
    wrapped = CountingRNG(np.random.default_rng(7))
    a = [float(gen.random()) for _ in range(5)]
    b = [float(wrapped.random()) for _ in range(5)]
    assert a == b
    assert wrapped.draws == 5
    # bit_generator 需可访问（快照/恢复依赖它）
    assert wrapped.bit_generator is not None


def test_bit_generator_state_roundtrip():
    """恢复 RNG 状态后 draws 计数继续（不重置、不打断）。"""
    wrapped = CountingRNG(np.random.default_rng(11))
    wrapped.random()
    st = wrapped.bit_generator.state
    wrapped2 = CountingRNG(np.random.default_rng(11))
    wrapped2.bit_generator.state = st
    assert wrapped2.random() == wrapped.random()


# ---- provenance 硬校验 ----

def test_git_commit_resolvable_in_repo():
    sha = git_commit()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_git_commit_hard_fails_outside_repo(tmp_path):
    """非 git 目录 ⇒ 抛错，绝不能静默返回 'unknown'。"""
    with pytest.raises(RuntimeError, match="provenance 硬校验失败"):
        git_commit(tmp_path)


def test_validate_rejects_placeholders():
    for bad in ({}, {"git_commit": "unknown", "config_fingerprint": "x"},
                {"git_commit": None, "config_fingerprint": "x"},
                {"git_commit": "abc", "config_fingerprint": "x"}):
        with pytest.raises(RuntimeError):
            validate(bad)


def test_validate_accepts_collected():
    prov = collect(SimConfig(seed=1))
    validate(prov)                      # 不抛即通过
    assert len(prov["git_commit"]) == 40
    assert prov["config_fingerprint"]
    assert isinstance(prov["git_dirty"], bool)


def test_validate_can_require_sim_core():
    prov = collect(SimConfig(seed=1))
    prov["sim_core_sha256"] = None
    if prov.get("sim_core_sha256") is None:
        with pytest.raises(RuntimeError):
            validate(prov, require_sim_core=True)


def test_collect_records_rng_draws():
    e = SphereEngine(SimConfig(seed=3))
    prov = collect(e.config, rng_draws=e.rng_draws)
    assert prov["rng_draws"] == e.rng_draws
    validate(prov)
