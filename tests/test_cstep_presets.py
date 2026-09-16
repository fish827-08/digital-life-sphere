"""C 步（R109 §五 本地直通车）预设的回归测试。

覆盖：
- 三段预设的**任务数与臂构成**（cstep1a 18 / cstep1b 6 / cstep2rand 6）；
- **变体（`variants`）机制**：三条臂参数互异、`--out` 命名正确、开关齐全；
- 🔴 **快照目录隔离**（`_rerun_logs/cstep_snap`）—— 防"与历史批重名 ⇒ 静默续跑"；
- 校准旗与科学臂的**互斥**：零模型/主臂不得带 `--calibration-arm`（R100 条件 5 干净面）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import batch_runner as br  # noqa: E402

CSTEP = ("cstep1a", "cstep1b", "cstep2rand")
SNAP_DIR = "_rerun_logs/cstep_snap"


def _by_name(runs) -> dict:
    return {r.name: r for r in runs}


def _flag(cmd: list[str], flag: str) -> str | None:
    """取 `--flag value` 的 value（flag 不存在 ⇒ None）。"""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# ------------------------------------------------------------ 任务数与命名

@pytest.mark.parametrize("name,n", [("cstep1a", 18), ("cstep1b", 6), ("cstep2rand", 6)])
def test_cstep_preset_run_counts(name, n):
    runs = br.preset_runs(name, Path("."))
    assert len(runs) == n, f"{name} 应为 {n} run"
    seeds = sorted({int(r.name.rsplit("_s", 1)[1]) for r in runs})
    assert seeds == [42, 43, 44, 45, 46, 47], f"{name}: seed 须为 42–47"


def test_cstep1a_has_three_distinct_arms_by_name_and_template():
    runs = _by_name(br.preset_runs("cstep1a", Path(".")))
    for arm in ("oracle_m1.3", "oracle_m1.0", "zero"):
        assert f"{arm}_s42" in runs, f"缺臂 {arm}"
    # 模板路径分段（供判读脚本按目录归类）
    assert runs["oracle_m1.3_s42"].out.parent.as_posix().endswith("_rerun_logs/cstep1a")
    assert runs["zero_s47"].out.name == "zero_s47.csv"


# ------------------------------------------------------------ 变体机制

def test_variant_flags_are_arm_specific():
    """三条臂的**专属开关**必须各自正确（这是 variants 存在的理由）。"""
    runs = _by_name(br.preset_runs("cstep1a", Path(".")))
    m13, m10, zero = runs["oracle_m1.3_s42"], runs["oracle_m1.0_s42"], runs["zero_s42"]

    assert _flag(m13.cmd, "--arm") == "oracle"
    assert _flag(m13.cmd, "--gain-multiplier") == "1.3"
    assert "--calibration-arm" in m13.cmd

    assert _flag(m10.cmd, "--arm") == "oracle"
    assert _flag(m10.cmd, "--gain-multiplier") == "1.0"
    assert "--calibration-arm" in m10.cmd

    assert _flag(zero.cmd, "--arm") == "zero"
    assert "--gain-multiplier" not in zero.cmd          # 零模型不得传增益档
    assert "--calibration-arm" not in zero.cmd          # 零模型须留在科学面（条件 5）


def test_science_arms_never_carry_calibration_flag():
    """🔴 R100 条件 5 的**干净面**：`zero`（cstep1a）与 `main`（cstep1b）不得带校准旗。

    科学集若混入校准臂，判读时会被机器拒收（或更糟：静默污染）。
    """
    for name in ("cstep1a", "cstep1b", "cstep2rand"):
        for r in br.preset_runs(name, Path(".")):
            is_science = r.name.startswith(("zero_", "main_"))
            if is_science:
                assert "--calibration-arm" not in r.cmd, f"{name}/{r.name} 不该带校准旗"


def test_cstep2rand_is_random_signal_and_calibration():
    runs = _by_name(br.preset_runs("cstep2rand", Path(".")))
    r = runs["rand_m1.3_s42"]
    assert _flag(r.cmd, "--signal-mode") == "random"   # 条件 7：无信息信号
    assert _flag(r.cmd, "--gain-multiplier") == "1.3"
    assert "--calibration-arm" in r.cmd                # 校准臂（不进科学判定）


# ------------------------------------------------------------ 快照目录隔离

@pytest.mark.parametrize("name", CSTEP)
def test_cstep_all_use_dedicated_snapshot_dir(name):
    """🔴 三段一律用**专属**快照目录 —— 否则 `zero_s42`/`main_s42` 与 D-24 同名会静默续跑。"""
    for r in br.preset_runs(name, Path(".")):
        assert _flag(r.cmd, "--snapshot-dir") == SNAP_DIR, f"{name}/{r.name} 快照目录未隔离"


def test_cstep_snapshot_names_do_not_collide_with_d24_default_dir():
    """把 C 步的快照路径与默认目录对照：证明**不落在** `_rerun_logs/snap/`。"""
    for name in CSTEP:
        for r in br.preset_runs(name, Path(".")):
            snap = Path(_flag(r.cmd, "--snapshot-dir")) / f"{r.name}.snapshot.npz"
            assert not snap.as_posix().startswith("_rerun_logs/snap/")
            assert snap.as_posix().startswith("_rerun_logs/cstep_snap/")


# ------------------------------------------------------------ 共同口径

@pytest.mark.parametrize("name", CSTEP)
def test_cstep_common_protocol_values(name):
    """共同口径：60k / max_count=3240（R41）/ donation=1.0 / 每 5000 快照。"""
    for r in br.preset_runs(name, Path(".")):
        assert _flag(r.cmd, "--ticks") == "60000"
        assert _flag(r.cmd, "--max-count") == "3240"
        assert _flag(r.cmd, "--donation") == "1.0"
        assert _flag(r.cmd, "--snapshot-every") == "5000"
        assert _flag(r.cmd, "--mode") == "on"
