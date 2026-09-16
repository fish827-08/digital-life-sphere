"""F-R21 回归：**preset 生成的 CLI 必须是目标脚本真正支持的开关**。

缺陷原型（R108 立号）
---------------------
`r97cal` preset 的网格键写作 `gain_multiplier`（下划线），而 `expand()` 直接拼 `--{key}`
⇒ 生成 `--gain_multiplier`，但 `argparse` 注册的是 `--gain-multiplier`
⇒ **参数不被识别**，该 preset 事实上不可直接执行（云端只能绕行 shell；无数据损失）。

本测试把「preset 的 CLI 与脚本开关一致」**机器化**：对**每一个** preset，
用目标脚本的 `--help` 取出真实开关集，再断言 preset 展开出的每个 `--flag` 都在其中
⇒ 同类缺陷（任意 preset、任意键名笔误）在 CI 即被拦，不必等云端跑挂。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import batch_runner as br  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
_FLAG_RE = re.compile(r"(--[A-Za-z0-9][-A-Za-z0-9]*)")
_HELP_CACHE: dict[str, set[str]] = {}


def _script_flags(script: str) -> set[str]:
    """目标脚本支持的开关全集（取自 `--help`）。"""
    if script not in _HELP_CACHE:
        out = subprocess.run([PY, str(ROOT / script), "--help"],
                             capture_output=True, text=True, cwd=str(ROOT))
        text = (out.stdout or "") + (out.stderr or "")
        _HELP_CACHE[script] = set(_FLAG_RE.findall(text))
        # 兜底：`--help` 本身
        _HELP_CACHE[script] |= {"--help", "-h"}
    return _HELP_CACHE[script]


# ------------------------------------------------------------ 单测：键名转换

def test_cli_flag_translates_underscore_to_hyphen():
    """F-R21 本体：键用下划线、CLI 用短横线。"""
    assert br.cli_flag("gain_multiplier") == "--gain-multiplier"
    assert br.cli_flag("max_count") == "--max-count"
    assert br.cli_flag("snapshot-every") == "--snapshot-every"   # 已是短横线 ⇒ 幂等
    assert br.cli_flag("arm") == "--arm"


def test_expand_keeps_template_placeholder_but_fixes_cli():
    """模板占位符仍用**原始键**（下划线），CLI 用短横线 —— 两者不得混淆。"""
    runs = br.expand(["gain_multiplier=1.0,1.3", "seed=42"],
                     ["mode=on", "max-count=3240"],
                     template="_tmp_test/m{gain_multiplier}_s{seed}.csv",
                     script="experiments/a4_verify_capacity.py",
                     workdir=Path("."))
    assert runs, "应展开出 run"
    argv = runs[0].cmd
    assert "--gain-multiplier" in argv and "--gain_multiplier" not in argv
    assert "--max-count" in argv
    # 模板用**原始键**（下划线）取值 ⇒ out 文件名里能看到 m 档与 seed
    assert runs[0].out.name == "m1.0_s42.csv"


def test_expand_grid_values_are_passed_positionally_after_flag():
    runs = br.expand(["seed=42,43"], [], "t_{seed}.csv",
                     "experiments/a4_verify_capacity.py", Path("."))
    assert [r.cmd[-1] for r in runs] == ["42", "43"]


# ------------------------------------------------------------ 全 preset 一致性（核心）

@pytest.mark.parametrize("name", sorted(br.PRESETS))
def test_preset_cli_flags_are_all_accepted_by_target_script(name):
    """🔴 **全 preset 排查**（机器化）：每个 `--flag` 都必须在目标脚本 `--help` 里出现。"""
    p = br.PRESETS[name]
    runs = br.expand(p["grid"], p["fixed"], p["template"], p["script"], Path("."))
    assert runs, f"{name}: 未展开出任何 run"
    valid = _script_flags(p["script"])
    bad: set[str] = set()
    for r in runs:
        for a in r.cmd:
            if a.startswith("--"):
                bad |= {a} - valid
    assert not bad, (
        f"{name}: 生成了目标脚本不支持的开关 {sorted(bad)}（F-R21 同型：键名与 "
        f"argparse 不一致）⇒ 该 preset 事实上不可直接执行"
    )


def test_all_presets_have_required_keys():
    """结构自检：每个 preset 必备 4 键（防手写 preset 漏字段 ⇒ 展开时 KeyError）。"""
    for name, p in br.PRESETS.items():
        assert {"script", "grid", "fixed", "template"} <= set(p), f"{name} 缺键"
        assert isinstance(p["grid"], list) and p["grid"], f"{name}: grid 须非空 list"


def test_r97cal_preset_is_now_runnable_shape():
    """F-R21 的**定向回归**：`r97cal` 必须生成 `--gain-multiplier`（而非下划线版）。"""
    p = br.PRESETS["r97cal"]
    runs = br.expand(p["grid"], p["fixed"], p["template"], p["script"], Path("."))
    assert len(runs) == 18, "r97cal = 3 档 m × 6 seed = 18 run"
    assert all("--gain-multiplier" in r.cmd for r in runs)
    assert all("--calibration-arm" in r.cmd for r in runs), "校准臂旗必须在（条件 5 依据）"
    assert not any("--gain_multiplier" in r.cmd for r in runs)
    # 模板里的 m 值必须来自网格（1.0 / 1.3 / 1.5）
    ms = sorted({r.out.name.split("_")[0] for r in runs})
    assert ms == ["m1.0", "m1.3", "m1.5"]
