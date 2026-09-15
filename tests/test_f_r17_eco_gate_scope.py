"""F-R17：`eco_gate_pass` 的**适用域**必须随数据走（判据适用域缺失）。

背景（2026-09-15 本人实测踩到）：`eco_gate_pass` 的判据含 `last >= 10000` 硬编码
（为 60k 批定义）⇒ **短程批恒 False**。字段本身没有适用域说明，于是
`_rerun_logs/d27` 那批 8000-tick 的 12/12 False 被我第一版拟合脚本当成"生态门不来"
⇒ **险些写成"生态崩溃"**（实际：灭绝 0/12、N 中位 2773）。

修法：summary 增 `eco_gate_scope`（该数据属于哪个 tick 批）与
`eco_gate_applicable`（本 tick 下该门是否适用），使"False 是因为不合格"与
"False 是因为不适用"**可区分**。

同族：F-R14（R78-3 对 zero 臂误报）／R98 扫描 —— **判据必须标注适用域**。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run_a4(out: Path, ticks: int) -> dict:
    snap = out.parent / "snap"
    rc = subprocess.call(
        [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
         "--mode", "on", "--arm", "main", "--seed", "42", "--ticks", str(ticks),
         "--max-count", "300", "--snapshot-every", "0",
         "--snapshot-dir", str(snap), "--out", str(out)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert rc == 0, f"a4 rc={rc}"
    return json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))


def test_short_batch_marks_eco_gate_inapplicable(tmp_path):
    """短程批（< 10000 tick）⇒ `eco_gate_applicable=False`，且 scope 如实标注。"""
    d = _run_a4(tmp_path / "short.csv", 300)
    r = d["result"]
    assert r["eco_gate_applicable"] is False
    assert r["eco_gate_scope"] == "300 tick 批"
    # 短程下 eco_gate_pass 必为 False —— 但它**不代表**生态不合格（适用域外）
    assert r["eco_gate_pass"] is False


def test_eco_gate_applicable_is_derived_from_ticks_target(tmp_path):
    """适用域判据 = `ticks_target >= 10000`（与 60k 批口径对齐的边界）。"""
    src = (ROOT / "experiments" / "a4_verify_capacity.py").read_text(encoding="utf-8")
    assert "eco_gate_applicable" in src and "args.ticks >= 10000" in src, (
        "适用域必须由 ticks_target 推导，不许手写 True/False"
    )
