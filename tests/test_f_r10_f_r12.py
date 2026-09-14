"""F-R10 / F-R12 回归测试（2026-09-15 D-24 实测暴露的两个收尾/续跑缺陷）。

**F-R10**（rc=1 假失败）：`a4_verify_capacity.py` 收尾删除 `*.progress.json` 被环境
safe-delete 守卫 fail-closed 拒绝 ⇒ 子进程退出码 1，而 summary 已先写、数据无损。
修法：①a4 收尾**不删**，改写 `{"status":"complete"}` 标记；②`batch_runner.poll()`
判据放宽为「summary 合法且 mtime 不早于本次启动」⇒ done（附 ⚠️ rc≠0 注记）。

**F-R12**（续批 CSV 重复段）：续跑直接 `open("a")` 追加 ⇒ 多轮续批重复写同一段 tick。
修法：续跑前**截断到 start_tick**。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.batch_runner import Run, poll, summary_ok  # noqa: E402

PY = sys.executable


# --------------------------------------------------------------- F-R10

def _write_summary(p: Path, ticks: int = 60000) -> None:
    p.write_text(
        json.dumps(
            {
                "manifest": {"finished": "x"},
                "switches": {"ticks_target": ticks},
                "result": {"final_N": 10, "final_tick": ticks, "eco_gate_pass": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class _FakeProc:
    def __init__(self, rc: int) -> None:
        self._rc = rc

    def poll(self):
        return self._rc


def test_poll_done_on_rc0_fresh_summary(tmp_path):
    r = Run(name="a", cmd=[], out=tmp_path / "a.csv", summary=tmp_path / "a.summary.json")
    _write_summary(r.summary)
    r._t0 = time.time() - 5
    r.proc = _FakeProc(0)
    poll([r])
    assert r.status == "done" and r.note == ""


def test_poll_done_on_rc1_with_fresh_summary(tmp_path):
    """F-R10 核心：rc=1 + summary 完整且为本轮产出 ⇒ **判 done**（不再重试）。"""
    r = Run(name="b", cmd=[], out=tmp_path / "b.csv", summary=tmp_path / "b.summary.json")
    _write_summary(r.summary)
    r._t0 = time.time() - 5
    r.proc = _FakeProc(1)
    poll([r])
    assert r.status == "done", "rc=1 假失败必须判 done"
    assert "F-R10" in r.note and "rc=1" in r.note


def test_poll_failed_on_stale_summary(tmp_path):
    """防"读到上一轮旧 summary"：mtime 早于本次启动 ⇒ 仍算 failed。"""
    r = Run(name="c", cmd=[], out=tmp_path / "c.csv", summary=tmp_path / "c.summary.json")
    _write_summary(r.summary)
    old = time.time() - 3600
    os.utime(r.summary, (old, old))
    r._t0 = time.time()
    r.proc = _FakeProc(1)
    poll([r])
    assert r.status == "failed"
    assert "旧文件" in r.note


def test_poll_failed_when_summary_missing(tmp_path):
    r = Run(name="d", cmd=[], out=tmp_path / "d.csv", summary=tmp_path / "d.summary.json")
    r._t0 = time.time()
    r.proc = _FakeProc(1)
    poll([r])
    assert r.status == "failed"


# ------------------------------------------------ a4 收尾不再删除（F-R10 集成）

def test_a4_writes_completion_marker_instead_of_deleting(tmp_path):
    out = tmp_path / "t.csv"
    snap = tmp_path / "snap"
    snap.mkdir()
    rc = subprocess.call(
        [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
         "--mode", "on", "--arm", "main", "--seed", "42", "--ticks", "400",
         "--max-count", "300", "--snapshot-every", "0",
         "--snapshot-dir", str(snap), "--out", str(out)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert rc == 0, f"a4 应 rc=0（F-R10 后不得因删除失败）: rc={rc}"
    prog = out.with_suffix(".progress.json")
    assert prog.exists(), "progress.json 必须保留（改写为完成标记）"
    assert json.loads(prog.read_text(encoding="utf-8"))["status"] == "complete"
    assert out.with_suffix(".summary.json").exists()


# -------------------------------------------- F-R12：续跑 CSV 按 tick 幂等（集成）

def _ticks_of(p: Path) -> list[int]:
    import csv
    return [int(r["tick"]) for r in csv.DictReader(p.open(encoding="utf-8"))]


@pytest.mark.parametrize("rounds", [2, 3])
def test_two_round_resume_csv_has_no_duplicate_ticks(tmp_path, rounds):
    """F-R12 核心：多轮续批后 CSV 必须**单调且无重复 tick**。"""
    out = tmp_path / "r.csv"
    snap = tmp_path / "snap"
    snap.mkdir()
    for i in range(1, rounds + 1):
        rc = subprocess.call(
            [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
             "--mode", "on", "--arm", "main", "--seed", "42",
             "--ticks", str(200 * i),
             "--max-count", "300", "--log-interval", "100",
             "--snapshot-every", "100", "--snapshot-dir", str(snap),
             "--out", str(out)],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        assert rc == 0, f"第 {i} 轮 rc={rc}"
    ticks = _ticks_of(out)
    assert ticks, "CSV 不应为空"
    assert len(ticks) == len(set(ticks)), f"存在重复 tick（F-R12 未修好）: {ticks}"
    assert all(b > a for a, b in zip(ticks, ticks[1:])), f"非单调: {ticks}"
    assert ticks[-1] == 200 * rounds
