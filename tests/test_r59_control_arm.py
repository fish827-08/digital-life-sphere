# -*- coding: utf-8 -*-
"""R59/F-R9 回归测试：control 臂必须关码本（arbitrary_codebook=False，codebook_conv ≡ 1.0）。

背景：2026-09-13 D-24 首跑发现 control 与 main 同配置同轨迹——--arm 映射缺
"control → 关码本"，batch grid 只传 arm 时 control 落到 --codebook 默认值 1。

本测试跑两个超短 run（300 tick，秒级），断言：
  1. control 臂 summary 的 switches.arbitrary_codebook == False
  2. control 臂 csv 的 codebook_conv 列 ≡ 1.0（±1e-9）
  3. main 臂 summary 的 switches.arbitrary_codebook == True（对照不被误伤）
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "a4_verify_capacity.py"
PY = sys.executable


def _run_arm(arm: str, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(SCRIPT),
        "--mode", "on", "--seed", "42", "--ticks", "300",
        "--log-interval", "100",
        "--max-count", "500",
        "--arm", arm,
        "--snapshot-every", "0",
        "--out", str(out),
    ]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{arm} run failed: {r.stderr[-500:]}"
    summary_path = out.with_suffix(".summary.json")
    assert summary_path.exists(), f"{arm} summary missing"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _csv_col(path: Path, col: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [float(row[col]) for row in csv.DictReader(f) if row.get("tick")]


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r59")
    control = _run_arm("control", tmp / "control_s99.csv")
    main = _run_arm("main", tmp / "main_s99.csv")
    return control, main, tmp


def test_control_arm_codebook_off(arms):
    control, _, _ = arms
    sw = (control.get("manifest", {}).get("switches", {})
          or control.get("switches", {}))
    assert sw.get("arbitrary_codebook") is False, \
        f"control 臂码本未关: {sw}"


def test_control_arm_codebook_conv_constant_one(arms):
    control, _, tmp = arms
    vals = _csv_col(tmp / "control_s99.csv", "codebook_conv")
    assert vals, "control csv 无 codebook_conv 列"
    assert all(abs(v - 1.0) <= 1e-9 for v in vals), \
        f"control 臂 codebook_conv 非恒 1.0: {vals}"


def test_main_arm_codebook_still_on(arms):
    """对照：main 臂码本仍开（防修复误伤主臂）。"""
    _, main, _ = arms
    sw = (main.get("manifest", {}).get("switches", {})
          or main.get("switches", {}))
    assert sw.get("arbitrary_codebook") is True, \
        f"main 臂码本被误关: {sw}"


def test_control_differs_from_main(arms):
    """control 与 main 的轨迹必须不同（F-R9 症状反向断言）。"""
    control, main, tmp = arms
    c = _csv_col(tmp / "control_s99.csv", "g15")
    m = _csv_col(tmp / "main_s99.csv", "g15")
    n = min(len(c), len(m))
    assert n > 0
    assert c[:n] != m[:n], "control 与 main 轨迹完全一致 ⇒ 配置仍然相同！"
